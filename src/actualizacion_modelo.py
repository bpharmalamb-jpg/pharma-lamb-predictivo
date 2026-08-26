# -*- coding: utf-8 -*-
"""Operacion del tablero: validar Excel de ventas reales, actualizar historial,
generar features G2, reentrenar Random Forest RF_09 y guardar artefactos
operacionales versionados en PROJECT_ROOT/artifacts_operacion/.

Separacion de responsabilidades:
- src/forecasting.py  -> carga/validacion/pronostico (estable, no entrenar).
- este modulo          -> preparacion de datos y reentrenamiento operacional.

Nunca modifica artifacts/ (investigacion congelada) ni notebooks.
El modelo operacional es el MISMO Random Forest RF_09 de la investigacion
(hiperparmetros exactos), reajustado unicamente con el historial real
actualizado. Sin tuning, sin XGBoost, sin clima, sin recalculo de metricas
(evaluation_reference y forecast_reference permanecen congeladas).
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from src.forecasting import (
    ARTIFACT_DIR,
    PROJECT_ROOT,
    _construir_calendario,
    cargar_artefactos_produccion,
    construir_calendario_siguiente_mes,
    generar_pronostico_siguiente_mes,
    verificar_integridad_artefactos,
)

INVESTIGACION_DIR = PROJECT_ROOT / "artifacts_investigacion"
OPERACION_DIR = PROJECT_ROOT / "artifacts_operacion"
VERSIONES_DIR = OPERACION_DIR / "versiones"

NOMBRES_MESES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

COLUMNAS_REQUERIDAS = ["Fecha Venta", "Nombre Producto", "Cantidad"]
COLUMNAS_OPCIONALES = ["Top", "Comprobante Venta", "Presentacion", "Categoria"]
HOJA_GENERAL = "Rotacion"


def _cargar_referencia_investigacion() -> tuple[dict, dict]:
    """Bundle/config de investigacion (artifacts/ canonicos), solo lectura."""
    bundle, config, _ = cargar_artefactos_produccion(ARTIFACT_DIR)
    return bundle, config


def cargar_referencia_operacional_actual() -> tuple[dict, dict]:
    """Carga el modelo operacional activo o, si no existe, el de investigacion."""
    directorio_activo = directorio_operativo_activo()
    if directorio_activo is not None:
        bundle, config, _ = cargar_artefactos_produccion(directorio_activo)
        return bundle, config
    return _cargar_referencia_investigacion()


def _leer_excel(archivo) -> tuple[pd.DataFrame, str, int]:
    """Lee el Excel y detecta la fila de encabezado con las columnas reales.

    Devuelve (df_limpio, nombre_hoja, fila_encabezado).
    """
    if isinstance(archivo, (bytes, bytearray)):
        archivo = io.BytesIO(archivo)
    excel = pd.ExcelFile(archivo)
    hoja = HOJA_GENERAL if HOJA_GENERAL in excel.sheet_names else excel.sheet_names[0]
    bruto = pd.read_excel(excel, sheet_name=hoja, header=None)

    fila_encabezado = None
    for fila in range(min(9, len(bruto))):
        valores = [str(v) for v in bruto.iloc[fila].tolist() if pd.notna(v)]
        if all(nombre in valores for nombre in COLUMNAS_REQUERIDAS):
            fila_encabezado = fila
            break
    if fila_encabezado is None:
        raise ValueError(
            "El archivo cargado no contiene las columnas requeridas "
            "(Fecha Venta, Nombre Producto, Cantidad) en las primeras filas."
        )

    df = pd.read_excel(excel, sheet_name=hoja, header=fila_encabezado)
    df = df[[c for c in COLUMNAS_REQUERIDAS + COLUMNAS_OPCIONALES if c in df.columns]]
    df = df.dropna(how="all").copy()
    return df, hoja, fila_encabezado


def _parsear_fechas(serie) -> tuple[pd.Series, int]:
    """Convierte Fecha Venta a datetime normalizado; reporta invalidas."""
    fechas = pd.to_datetime(serie, format="%d/%m/%Y", errors="coerce")
    if fechas.notna().mean() < 0.9:
        fechas = pd.to_datetime(serie, dayfirst=True, errors="coerce")
    fechas = pd.DatetimeIndex(fechas).normalize()
    invalidas = int(pd.isna(fechas).sum())
    return fechas, invalidas


def _parsear_cantidades(serie) -> tuple[pd.Series, int]:
    cantidades = pd.to_numeric(serie, errors="coerce")
    invalidas = int(pd.isna(cantidades).sum())
    return cantidades, invalidas


def validar_excel(archivo, bundle_referencia: dict | None = None) -> dict:
    """Valida un Excel de ventas reales y devuelve un resumen completo.

    Comprueba: Excel valido, columnas requeridas, fechas convertibles,
    cantidades numericas, nombres validos, coherencia cronologica,
    ultima fecha real y duplicados potenciales (se advierten, NO se eliminan).

    Devuelve dict con `valido`, `ventas` (filas limpias) y `resumen` para
    mostrar al usuario antes de actualizar el modelo.
    """
    if bundle_referencia is None:
        bundle_referencia, _ = _cargar_referencia_investigacion()
    top10 = list(bundle_referencia["top10_final"])

    df, hoja, fila_encabezado = _leer_excel(archivo)
    n_original = len(df)

    advertencias = []
    errores = []
    if not df["Nombre Producto"].notna().all():
        advertencias.append(
            f"{int(df['Nombre Producto'].isna().sum())} filas sin Nombre Producto; "
            "se excluyen del panel."
        )
    df = df[df["Nombre Producto"].notna()].copy()
    df["Nombre Producto"] = df["Nombre Producto"].astype(str).str.strip()
    df = df[df["Nombre Producto"] != ""].copy()

    fechas, n_fechas_invalidas = _parsear_fechas(df["Fecha Venta"])
    df["Fecha Venta"] = fechas
    cantidades, n_cantidades_invalidas = _parsear_cantidades(df["Cantidad"])
    df["Cantidad"] = cantidades

    validas = df["Fecha Venta"].notna() & df["Cantidad"].notna()
    n_descartadas = int((~validas).sum())
    if n_fechas_invalidas:
        errores.append(
            f"{n_fechas_invalidas} fechas no pudieron convertirse a fecha valida."
        )
    if n_cantidades_invalidas:
        errores.append(
            f"{n_cantidades_invalidas} cantidades no son numericas."
        )
    if n_descartadas:
        advertencias.append(
            f"{n_descartadas} filas sin fecha o cantidad valida se excluyen del panel."
        )
    ventas = df[validas].copy()

    fecha_min = ventas["Fecha Venta"].min() if len(ventas) else None
    fecha_max = ventas["Fecha Venta"].max() if len(ventas) else None
    if not len(ventas):
        errores.append("El archivo no contiene filas validas para actualizar el historial.")

    if fecha_min is not None and fecha_min < pd.Timestamp("2024-01-02"):
        advertencias.append(
            "El archivo incluye fechas anteriores al inicio del historial de "
            "investigacion (2024-01-02); el panel conserva la cronologia completa."
        )
    if fecha_max is not None and fecha_max > pd.Timestamp.now().normalize():
        advertencias.append("El archivo incluye fechas posteriores a hoy.")

    max_real_actual = pd.Timestamp(bundle_referencia["historial_top10"]["Fecha Venta"].max())
    if fecha_max is not None and fecha_max <= max_real_actual:
        errores.append(
            "La ultima fecha real del archivo no supera al historial actual "
            f"({max_real_actual.date()}); no se incorporarian datos nuevos."
        )
    if fecha_min is not None:
        min_real_actual = pd.Timestamp(bundle_referencia["historial_top10"]["Fecha Venta"].min())
        if fecha_min > min_real_actual:
            errores.append(
                "El archivo debe contener el historial completo actualizado. "
                f"La fecha inicial encontrada ({fecha_min.date()}) es posterior "
                f"al inicio del historial actual ({min_real_actual.date()})."
            )
        fechas_actuales = pd.DatetimeIndex(
            bundle_referencia["historial_top10"]["Fecha Venta"].drop_duplicates()
        ).normalize()
        fechas_archivo = pd.DatetimeIndex(ventas["Fecha Venta"].drop_duplicates()).normalize()
        fechas_faltantes = fechas_actuales.difference(fechas_archivo)
        if len(fechas_faltantes):
            errores.append(
                "El archivo debe ser un historial completo actualizado. "
                f"Faltan {len(fechas_faltantes)} fechas de actividad ya presentes "
                "en el modelo actual."
            )

    productos_archivo = set(ventas["Nombre Producto"].unique()) if len(ventas) else set()
    fuera_top10 = sorted(productos_archivo - set(top10))
    if fuera_top10:
        advertencias.append(
            f"{len(fuera_top10)} medicamentos del archivo no pertenecen al Top10 "
            "de investigacion; quedan fuera del modelo (por ejemplo: "
            + ", ".join(fuera_top10[:3]) + ")."
        )
    top10_sin_ventas = [p for p in top10 if p not in productos_archivo]
    if top10_sin_ventas:
        advertencias.append(
            f"{len(top10_sin_ventas)} medicamentos del Top10 no registran ventas "
            "en el archivo; se incluyen con demanda cero en las fechas activas."
        )

    n_duplicados = 0
    if len(ventas) and "Comprobante Venta" in ventas.columns:
        claves = ventas.duplicated(
            subset=["Comprobante Venta", "Nombre Producto"], keep=False
        ).sum()
        if claves:
            advertencias.append(
                f"{claves} filas comparten comprobante y medicamento (posibles "
                "transacciones repetidas). No se eliminan automaticamente."
            )
            n_duplicados = int(claves)

    resumen = {
        "hoja": hoja,
        "fila_encabezado": fila_encabezado,
        "n_filas_originales": n_original,
        "n_filas_validas": len(ventas),
        "n_descartadas": n_descartadas,
        "n_fechas_invalidas": n_fechas_invalidas,
        "n_cantidades_invalidas": n_cantidades_invalidas,
        "fecha_min": fecha_min,
        "fecha_max": fecha_max,
        "n_productos_archivo": len(productos_archivo),
        "productos_fuera_top10": fuera_top10,
        "top10_sin_ventas": top10_sin_ventas,
        "n_duplicados_potenciales": n_duplicados,
        "columnas_presentes": [c for c in COLUMNAS_REQUERIDAS + COLUMNAS_OPCIONALES if c in df.columns],
        "advertencias": advertencias,
        "errores": errores,
    }
    valido = len(ventas) > 0 and not errores
    return {"valido": valido, "resumen": resumen, "ventas": ventas, "errores": errores}


def integrar_historial(
    ventas: pd.DataFrame, top10: list[str] | None = None
) -> pd.DataFrame:
    """Panel Top10 con demanda diaria real (misma logica que la investigacion).

    Fechas activas = fechas con venta real en el archivo (lunes a sabado).
    Dentro de cada fecha activa, los pares fecha x producto sin venta se
    completan con 0.0. Los dias sin actividad NO se agregan al panel.
    """
    if top10 is None:
        bundle_referencia, _ = _cargar_referencia_investigacion()
        top10 = list(bundle_referencia["top10_final"])

    fechas_actividad = pd.DatetimeIndex(sorted(ventas["Fecha Venta"].unique())).normalize()
    demanda_observada = (
        ventas.groupby(["Fecha Venta", "Nombre Producto"], as_index=False)["Cantidad"]
        .sum()
        .rename(columns={"Cantidad": "Demanda"})
    )
    grilla = pd.MultiIndex.from_product(
        [fechas_actividad, top10], names=["Fecha Venta", "Nombre Producto"]
    )
    panel = (
        demanda_observada.set_index(["Fecha Venta", "Nombre Producto"])
        .reindex(grilla)
        .reset_index()
    )
    panel["Demanda"] = panel["Demanda"].fillna(0.0).astype(float)
    orden_productos = {p: i for i, p in enumerate(top10)}
    panel["_orden"] = panel["Nombre Producto"].map(orden_productos)
    panel = panel.sort_values(["Fecha Venta", "_orden"]).drop(columns="_orden").reset_index(drop=True)
    return panel[["Fecha Venta", "Nombre Producto", "Demanda"]]


def construir_features_g2(panel: pd.DataFrame, grupo_g2: list[str]) -> pd.DataFrame:
    """Construye las 22 variables G2 (13 calendario + 6 LagObs + 3 LagCal).

    Logica identica a la investigacion: LagObs por producto (observaciones
    anteriores), LagCal por fecha calendario exacta (-7/-14/-28 dias) sin
    fallback; el resto queda NaN para el imputer.
    """
    data = panel[["Fecha Venta", "Nombre Producto", "Demanda"]].copy()
    data["Fecha Venta"] = pd.DatetimeIndex(data["Fecha Venta"]).normalize()
    data["Demanda"] = data["Demanda"].astype(float)
    data = data.sort_values(["Nombre Producto", "Fecha Venta"]).reset_index(drop=True)

    cal = _construir_calendario(data["Fecha Venta"]).reset_index(drop=True)
    data = pd.concat([data, cal], axis=1)

    g = data.groupby("Nombre Producto", sort=True)
    for n in [1, 2, 3, 6, 12, 30]:
        data[f"LagObs_{n}"] = g["Demanda"].shift(n)

    lookup = data.set_index(["Nombre Producto", "Fecha Venta"])["Demanda"].to_dict()
    for dias in [7, 14, 28]:
        data[f"LagCal_{dias}d"] = [
            lookup.get((p, f - pd.Timedelta(days=dias)))
            for p, f in zip(data["Nombre Producto"], data["Fecha Venta"])
        ]
    return data[grupo_g2 + ["Nombre Producto", "Demanda"]].copy()


def preparar_xy(
    panel: pd.DataFrame, grupo_g2: list[str]
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Construye preprocesador ajustado (imputer mediana + OHE) y X transformado."""
    features = construir_features_g2(panel, grupo_g2)
    X = features[grupo_g2 + ["Nombre Producto"]].copy()
    y = features["Demanda"].astype(float).to_numpy()

    imputer = SimpleImputer(strategy="median")
    num_imp = imputer.fit_transform(X[grupo_g2])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_enc = encoder.fit_transform(X[["Nombre Producto"]])
    nombres = list(grupo_g2) + list(encoder.get_feature_names_out(["Nombre Producto"]))
    Xt = np.hstack([num_imp, cat_enc])
    if Xt.shape[1] != 32 or not np.isfinite(Xt).all():
        raise ValueError("X transformado no tiene 32 columnas finitas")
    return {"imputer": imputer, "encoder": encoder, "nombres": nombres}, Xt, y


def _escribir_json(objeto, ruta: Path) -> None:
    with open(ruta, "w", encoding="utf-8") as fh:
        json.dump(objeto, fh, ensure_ascii=False, indent=2)
        fh.write("\n")


def _construir_manifest(directorio: Path) -> dict:
    registros = {}
    for nombre in ["bundle_produccion.joblib", "config_modelo.json"]:
        datos = (directorio / nombre).read_bytes()
        registros[nombre] = {
            "filename": nombre,
            "size_bytes": len(datos),
            "sha256": hashlib.sha256(datos).hexdigest(),
        }
    return {"schema_version": "1.0.0", "model_id": "RF09_G2_TOP10_2024_2025", "artifacts": registros}


def _respaldar_version_previa() -> str | None:
    """Guarda la version operacional actual como version historica."""
    if not (OPERACION_DIR / "manifest.json").is_file():
        return None
    if not VERSIONES_DIR.exists():
        VERSIONES_DIR.mkdir(parents=True, exist_ok=True)
    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    destino = VERSIONES_DIR / ("v_" + version)
    destino.mkdir(parents=True, exist_ok=True)
    for nombre in ["bundle_produccion.joblib", "config_modelo.json", "manifest.json"]:
        origen = OPERACION_DIR / nombre
        if origen.is_file():
            shutil.copy2(origen, destino / nombre)
    return str(destino)


def _reemplazar_operacion_desde_staging(staging_dir: Path) -> str | None:
    """Reemplaza artefactos operacionales despues de construirlos completos."""
    OPERACION_DIR.mkdir(parents=True, exist_ok=True)
    respaldo = _respaldar_version_previa()
    for nombre in ["bundle_produccion.joblib", "config_modelo.json", "manifest.json"]:
        shutil.copy2(staging_dir / nombre, OPERACION_DIR / nombre)
    return respaldo


def reentrenar_y_guardar(
    archivo, bundle_referencia: dict | None = None, config_referencia: dict | None = None
) -> dict:
    """Flujo operacional completo: validar -> integrar -> features -> RF_09 -> guardar.

    Escrituras SOLO bajo artifacts_operacion/ (con respaldo de la version
    anterior). Devuelve resumen con version, piezas y pronostico del siguiente
    mes (de referencia, sin tocar caches de la app).
    """
    if bundle_referencia is None or config_referencia is None:
        bundle_referencia, config_referencia = cargar_referencia_operacional_actual()

    resultado_validacion = validar_excel(archivo, bundle_referencia)
    if not resultado_validacion["valido"]:
        errores = resultado_validacion["resumen"].get("errores", [])
        detalle = " ".join(errores) if errores else "revise el resumen."
        raise ValueError("La validacion del archivo fallo: " + detalle)

    grupo_g2 = list(bundle_referencia["grupo_g2"])
    top10 = list(bundle_referencia["top10_final"])
    panel = integrar_historial(resultado_validacion["ventas"], top10)
    preprocesador, Xt, y = preparar_xy(panel, grupo_g2)

    rf = RandomForestRegressor(**config_referencia["model_params"])
    rf.fit(Xt, y)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_start = panel["Fecha Venta"].min().date().isoformat()
    training_end = panel["Fecha Venta"].max().date().isoformat()

    bundle_operacional = {
        "schema_version": bundle_referencia["schema_version"],
        "model_id": bundle_referencia["model_id"],
        "rf_produccion": rf,
        "preprocesador_produccion": preprocesador,
        "grupo_g2": grupo_g2,
        "top10_final": top10,
        "historial_top10": panel,
        "metadata": {
            "training_start": training_start,
            "training_end": training_end,
            "training_activity_dates": int(panel["Fecha Venta"].nunique()),
            "training_rows_top10": int(len(panel)),
            "transformed_features": 32,
            "categorical_feature": "Nombre Producto",
            "annual_cycle_divisor": 365.25,
            "clip_negative_predictions": True,
            "round_predictions": False,
            "calendar_policy": "lunes-sabado; domingos excluidos; cierres extraordinarios futuros no anticipados",
            "operacion_version": version,
            "reajuste_con_datos_hasta": training_end,
        },
    }

    config_operacional = json.loads(json.dumps(config_referencia, ensure_ascii=False))
    config_operacional["model_params"] = dict(config_referencia["model_params"])
    n_fechas_activas = int(panel["Fecha Venta"].nunique())
    n_productos_archivo = int(resultado_validacion["resumen"]["n_productos_archivo"])
    config_operacional["training"] = {
        "start": training_start,
        "end": training_end,
        "activity_dates": n_fechas_activas,
        "panel_rows_complete": n_fechas_activas * n_productos_archivo,
        "panel_rows_top10": bundle_operacional["metadata"]["training_rows_top10"],
        "transformed_features": 32,
        "categorical_levels": 10,
        "source": "archivo Excel de ventas reales cargado por el usuario",
        "features_rule": "LagObs con shift por producto; LagCal por fecha calendario exacta (-7/-14/-28 dias); NaN sin fallback",
    }
    config_operacional["operacion"] = {
        "version": version,
        "reajuste_con_datos_hasta": training_end,
        "procedimiento": "Reajuste de Random Forest RF_09 (hiperparametros congelados "
        "de la investigacion) con historial real actualizado; sin tuning, sin XGBoost, "
        "sin clima y sin recalculo de metricas de evaluacion.",
        "nota": "evaluation_reference y forecast_reference son de la investigacion "
        "y no se recalcularon.",
    }
    config_operacional["evaluation_reference"]["note"] += (
        " El modelo operacional se reajusta con datos reales actualizados "
        "sin recalcular estas metricas."
    )

    with tempfile.TemporaryDirectory(prefix="farmalab_operacion_") as tmp:
        staging = Path(tmp)
        joblib.dump(bundle_operacional, staging / "bundle_produccion.joblib", compress=3)
        _escribir_json(config_operacional, staging / "config_modelo.json")
        _escribir_json(_construir_manifest(staging), staging / "manifest.json")
        verificar_integridad_artefactos(staging)
        respaldo = _reemplazar_operacion_desde_staging(staging)

    pronostico_diario, pronostico_mensual = generar_pronostico_siguiente_mes(bundle_operacional)

    proximo_mes = construir_calendario_siguiente_mes(panel)
    anio_pm, mes_pm = proximo_mes.min().year, proximo_mes.min().month
    return {
        "version": version,
        "respaldo_version_previa": respaldo,
        "bundle": bundle_operacional,
        "config": config_operacional,
        "panel": panel,
        "proximo_mes": f"{NOMBRES_MESES[mes_pm]} {anio_pm}",
        "proximo_mes_anio": (anio_pm, mes_pm),
        "total_pronostico": float(pronostico_diario["Prediccion"].sum()),
        "pronostico_diario": pronostico_diario,
        "pronostico_mensual": pronostico_mensual,
        "validacion": resultado_validacion["resumen"],
    }


def directorio_operativo_activo() -> Path | None:
    """Directorio operacional si existe y es integro; si no, None (usar investigacion)."""
    if not (OPERACION_DIR / "manifest.json").is_file():
        return None
    try:
        verificar_integridad_artefactos(OPERACION_DIR)
        return OPERACION_DIR
    except ValueError:
        return None


__all__ = [
    "INVESTIGACION_DIR",
    "OPERACION_DIR",
    "COLUMNAS_REQUERIDAS",
    "cargar_referencia_operacional_actual",
    "validar_excel",
    "integrar_historial",
    "construir_features_g2",
    "preparar_xy",
    "reentrenar_y_guardar",
    "directorio_operativo_activo",
]
