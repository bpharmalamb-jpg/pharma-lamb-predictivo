# -*- coding: utf-8 -*-
"""Motor de inferencia para la aplicacion Streamlit (independiente de notebooks).

Carga los artefactos de produccion persistidos en PROJECT_ROOT/artifacts,
verifica su integridad SHA-256 contra el manifest, valida su estructura y
genera el pronostico recursivo del siguiente mes calendario planificado.

Reglas invariantes (etapas 5.3.2 / 6.1):
- No se entrena, no se hace fit ni fit_transform.
- El estado temporal es una copia; el bundle nunca se modifica.
- No se pronostican meses arbitrarios: solo el mes calendario inmediatamente
  posterior a la maxima fecha real del historial.
- No se calculan metricas de precision: enero 2026 no tiene demanda real.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
BUNDLE_PATH = ARTIFACT_DIR / "bundle_produccion.joblib"
CONFIG_PATH = ARTIFACT_DIR / "config_modelo.json"
MANIFEST_PATH = ARTIFACT_DIR / "manifest.json"

CLAVES_BUNDLE = frozenset(
    {
        "schema_version",
        "model_id",
        "rf_produccion",
        "preprocesador_produccion",
        "grupo_g2",
        "top10_final",
        "historial_top10",
        "metadata",
    }
)
COLUMNAS_HISTORIAL = ["Fecha Venta", "Nombre Producto", "Demanda"]
COLUMNAS_DIARIO = ["Fecha Venta", "Nombre Producto", "Prediccion"]
COLUMNAS_MENSUAL = ["Nombre Producto", "Pronostico_Mensual", "Ranking"]

REFERENCIA_ENERO_2026 = 8829.307412205015


def calcular_sha256(ruta: Path) -> str:
    """SHA-256 en hexadecimal (64 caracteres) del contenido de `ruta`."""
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def _leer_manifest(artifact_dir: Path) -> dict:
    with open(artifact_dir / "manifest.json", "r", encoding="utf-8") as fh:
        return json.load(fh)


def verificar_integridad_artefactos(artifact_dir: Path = ARTIFACT_DIR) -> dict:
    """Verifica existencia, hashes y tamanos de los artefactos contra el manifest.

    Si algo falla, lanza ValueError y NUNCA se llega a joblib.load.
    Devuelve un dict con el SHA-256 real y el manifest leido.
    """
    archivos = ["bundle_produccion.joblib", "config_modelo.json", "manifest.json"]
    faltantes = [n for n in archivos if not (artifact_dir / n).is_file()]
    if faltantes:
        raise ValueError("Faltan archivos de artefactos: " + ", ".join(faltantes))

    manifest = _leer_manifest(artifact_dir)
    if manifest.get("schema_version") != "1.0.0":
        raise ValueError("manifest.schema_version no es 1.0.0")
    if manifest.get("model_id") != "RF09_G2_TOP10_2024_2025":
        raise ValueError("manifest.model_id no es RF09_G2_TOP10_2024_2025")

    registrados = manifest.get("artifacts", {})
    if sorted(registrados.keys()) != ["bundle_produccion.joblib", "config_modelo.json"]:
        raise ValueError("manifest debe registrar solo bundle y config")

    reales = {}
    for nombre in ["bundle_produccion.joblib", "config_modelo.json"]:
        ruta = artifact_dir / nombre
        datos = ruta.read_bytes()
        reales[nombre] = {"size_bytes": len(datos), "sha256": hashlib.sha256(datos).hexdigest()}
        registro = registrados[nombre]
        if registro.get("size_bytes") != reales[nombre]["size_bytes"]:
            raise ValueError(
                "size_bytes no coincide para " + nombre
                + " (manifest=" + str(registro.get("size_bytes"))
                + ", real=" + str(reales[nombre]["size_bytes"]) + ")"
            )
        if registro.get("sha256") != reales[nombre]["sha256"]:
            raise ValueError("SHA-256 no coincide para " + nombre)

    return {
        "integro": True,
        "bundle_sha256": reales["bundle_produccion.joblib"]["sha256"],
        "config_sha256": reales["config_modelo.json"]["sha256"],
        "manifest": manifest,
    }


def validar_bundle_y_config(bundle: dict, config: dict, manifest: dict) -> bool:
    """Valida estructura del bundle y consistencia cruzada bundle/config/manifest.

    Lanza ValueError con el detalle completo si cualquier condicion falla.
    """
    fallas = []

    if set(bundle.keys()) != set(CLAVES_BUNDLE):
        fallas.append("bundle claves != las 8 esperadas")

    if bundle.get("schema_version") != "1.0.0":
        fallas.append("bundle.schema_version != 1.0.0")
    if bundle.get("model_id") != "RF09_G2_TOP10_2024_2025":
        fallas.append("bundle.model_id != RF09_G2_TOP10_2024_2025")
    if config.get("schema_version") != bundle.get("schema_version"):
        fallas.append("config.schema_version != bundle.schema_version")
    if manifest.get("schema_version") != bundle.get("schema_version"):
        fallas.append("manifest.schema_version != bundle.schema_version")
    if config.get("model_id") != bundle.get("model_id"):
        fallas.append("config.model_id != bundle.model_id")
    if manifest.get("model_id") != bundle.get("model_id"):
        fallas.append("manifest.model_id != bundle.model_id")

    grupo_g2 = bundle.get("grupo_g2")
    if not isinstance(grupo_g2, list) or len(grupo_g2) != 22 or len(set(grupo_g2)) != 22:
        fallas.append("grupo_g2 no es lista de 22 variables unicas")
    if list(config.get("feature_group", {}).get("variables", [])) != grupo_g2:
        fallas.append("config.feature_group.variables != bundle.grupo_g2")
    if config.get("feature_group", {}).get("name") != "G2":
        fallas.append("config.feature_group.name != G2")
    if config.get("feature_group", {}).get("total_variables") != 22:
        fallas.append("config.feature_group.total_variables != 22")

    top10 = bundle.get("top10_final")
    if not isinstance(top10, list) or len(top10) != 10:
        fallas.append("top10_final no tiene 10 productos")
    if config.get("top10", {}).get("count") != 10:
        fallas.append("config.top10.count != 10")
    if list(config.get("top10", {}).get("products", [])) != top10:
        fallas.append("config.top10.products != bundle.top10_final")

    historial = bundle.get("historial_top10")
    training = config.get("training", {})
    filas_esperadas = training.get("panel_rows_top10")
    fin_esperado = training.get("end")
    if not isinstance(filas_esperadas, int) or filas_esperadas <= 0:
        fallas.append("training.panel_rows_top10 ausente o no positivo")
    if historial is None or historial.shape != (filas_esperadas, 3):
        fallas.append(f"historial_top10.shape != ({filas_esperadas}, 3)")
    else:
        if list(historial.columns) != COLUMNAS_HISTORIAL:
            fallas.append("columnas historial != Fecha Venta/Nombre Producto/Demanda")
        if not pd.api.types.is_datetime64_any_dtype(historial["Fecha Venta"]):
            fallas.append("Fecha Venta del historial no es datetime")
        if historial["Nombre Producto"].nunique() != 10:
            fallas.append("productos unicos del historial != 10")
        if fin_esperado is not None:
            try:
                max_esperado = pd.Timestamp(fin_esperado)
                if historial["Fecha Venta"].max() != max_esperado:
                    fallas.append("max fecha historica != training.end (" + str(max_esperado.date()) + ")")
            except (TypeError, ValueError):
                fallas.append("training.end no es una fecha valida")

    if training.get("start") is not None:
        try:
            pd.Timestamp(training["start"])
        except (TypeError, ValueError):
            fallas.append("training.start no es una fecha valida")
    if training.get("end") is not None:
        try:
            pd.Timestamp(training["end"])
        except (TypeError, ValueError):
            fallas.append("training.end no es una fecha valida")
    if training.get("transformed_features") != 32:
        fallas.append("training.transformed_features != 32")

    reglas = config.get("forecast_rules", {})
    if reglas.get("recursive") is not True:
        fallas.append("forecast_rules.recursive != true")
    if reglas.get("lagobs") != [1, 2, 3, 6, 12, 30]:
        fallas.append("forecast_rules.lagobs != [1,2,3,6,12,30]")
    if reglas.get("lagcal_days") != [7, 14, 28]:
        fallas.append("forecast_rules.lagcal_days != [7,14,28]")
    if reglas.get("lagcal_exact_date") is not True:
        fallas.append("forecast_rules.lagcal_exact_date != true")
    if reglas.get("lagcal_fallback") is not None:
        fallas.append("forecast_rules.lagcal_fallback != null")
    if reglas.get("annual_cycle_divisor") != 365.25:
        fallas.append("forecast_rules.annual_cycle_divisor != 365.25")
    if reglas.get("negative_prediction_clipping") != 0.0:
        fallas.append("forecast_rules.negative_prediction_clipping != 0.0")
    if reglas.get("round_internal_predictions") is not False:
        fallas.append("forecast_rules.round_internal_predictions != false")

    preprocesador = bundle.get("preprocesador_produccion")
    if not isinstance(preprocesador, dict) or set(preprocesador.keys()) != {"imputer", "encoder", "nombres"}:
        fallas.append("preprocesador_produccion no es dict imputer/encoder/nombres")
    else:
        if len(preprocesador.get("nombres", [])) != 32:
            fallas.append("len(preprocesador.nombres) != 32")
        encoder = preprocesador.get("encoder")
        if encoder is None or not hasattr(encoder, "categories_"):
            fallas.append("encoder no ajustado (sin categories_)")
        elif len(encoder.categories_[0]) != 10:
            fallas.append("encoder categorias != 10")

    rf = bundle.get("rf_produccion")
    if rf is None:
        fallas.append("rf_produccion ausente")
    else:
        esperado = {
            "n_estimators": 500, "max_depth": 8, "min_samples_split": 2,
            "min_samples_leaf": 1, "max_features": "sqrt",
            "random_state": 42, "n_jobs": -1,
        }
        for k, v in esperado.items():
            if getattr(rf, k) != v:
                fallas.append("RF_09." + k + " != " + str(v))
        if not hasattr(rf, "estimators_") or len(rf.estimators_) != 500:
            fallas.append("len(rf.estimators_) != 500")

    if fallas:
        raise ValueError("Validacion de artefactos fallida:\n- " + "\n- ".join(fallas))
    return True


def cargar_artefactos_produccion(artifact_dir: Path = ARTIFACT_DIR) -> tuple[dict, dict, dict]:
    """Carga segura y validada de (bundle, config, manifest).

    Orden obligatorio: verificar integridad -> leer config -> joblib.load
    -> validar bundle/config/manifest -> devolver. No entrena nada.
    """
    integridad = verificar_integridad_artefactos(artifact_dir)
    with open(artifact_dir / "config_modelo.json", "r", encoding="utf-8") as fh:
        config = json.load(fh)
    bundle = joblib.load(artifact_dir / "bundle_produccion.joblib")
    validar_bundle_y_config(bundle, config, integridad["manifest"])
    return bundle, config, integridad["manifest"]


def obtener_siguiente_mes(historial: pd.DataFrame) -> tuple[int, int]:
    """Devuelve (anio, mes) del mes calendario inmediatamente posterior
    a la maxima fecha real del historial."""
    max_fecha = pd.Timestamp(historial["Fecha Venta"].max()).normalize()
    if max_fecha.month == 12:
        return max_fecha.year + 1, 1
    return max_fecha.year, max_fecha.month + 1


def construir_calendario_siguiente_mes(historial: pd.DataFrame) -> pd.DatetimeIndex:
    """Fechas planificadas del siguiente mes: lunes-sabado, domingos excluidos.

    No se excluyen feriados ni se inventan cierres extraordinarios.
    """
    anio, mes = obtener_siguiente_mes(historial)
    primer_dia = pd.Timestamp(anio, mes, 1)
    ultimo_dia = primer_dia + pd.offsets.MonthEnd(0)
    fechas = pd.DatetimeIndex(pd.date_range(primer_dia, ultimo_dia, freq="D"))
    fechas = pd.DatetimeIndex(fechas[fechas.dayofweek != 6]).normalize()
    return fechas


def _construir_calendario(fechas) -> pd.DataFrame:
    """Features de calendario identicos al pipeline de produccion (G2, 13 vars)."""
    s = pd.Series(pd.to_datetime(fechas)).reset_index(drop=True)
    cal = pd.DataFrame(index=s.index)
    cal["Anio"] = s.dt.year.astype(int)
    cal["Mes"] = s.dt.month.astype(int)
    cal["DiaDelMes"] = s.dt.day.astype(int)
    cal["DiaSemana"] = s.dt.dayofweek.astype(int)
    cal["DiaDelAno"] = s.dt.dayofyear.astype(int)
    cal["SemanaISO"] = s.dt.isocalendar().week.astype(int)
    cal["Trimestre"] = s.dt.quarter.astype(int)
    cal["Mes_Sin"] = np.sin(2 * np.pi * cal["Mes"] / 12)
    cal["Mes_Cos"] = np.cos(2 * np.pi * cal["Mes"] / 12)
    cal["DiaSemana_Sin"] = np.sin(2 * np.pi * cal["DiaSemana"] / 7)
    cal["DiaSemana_Cos"] = np.cos(2 * np.pi * cal["DiaSemana"] / 7)
    cal["DiaDelAno_Sin"] = np.sin(2 * np.pi * cal["DiaDelAno"] / 365.25)
    cal["DiaDelAno_Cos"] = np.cos(2 * np.pi * cal["DiaDelAno"] / 365.25)
    return cal


def transformar_filas_g2(preprocesador: dict, filas: pd.DataFrame, grupo_g2: list[str]) -> np.ndarray:
    """Transforma filas G2: imputer.transform + encoder.transform, sin fit.

    Devuelve un arreglo (n_filas, 32). El NaN de LagCal lo resuelve
    unicamente el SimpleImputer ya ajustado.
    """
    num = filas[grupo_g2].copy()
    cat = filas[["Nombre Producto"]].copy()
    Xt = np.hstack(
        [
            preprocesador["imputer"].transform(num),
            preprocesador["encoder"].transform(cat),
        ]
    )
    if not np.isfinite(Xt).all():
        raise ValueError("X transformado contiene NaN/inf")
    return Xt


def generar_pronostico_siguiente_mes(
    bundle: dict, auditar_estado: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Pronostico recursivo del siguiente mes (diario + mensual).

    Por cada fecha futura se construyen las 10 filas contra el mismo estado
    pre-fecha; recien DESPUES de predecir los 10 productos se incorporan las
    predicciones al estado temporal (anti-leakage entre productos).
    Con `auditar_estado=True` devuelve ademas la auditoria de tamaños de estado.
    """
    historial = bundle["historial_top10"]
    productos = list(bundle["top10_final"])
    preprocesador = bundle["preprocesador_produccion"]
    modelo = bundle["rf_produccion"]
    grupo_g2 = list(bundle["grupo_g2"])

    fechas_futuras = construir_calendario_siguiente_mes(historial)
    origen = pd.Timestamp(fechas_futuras.min()).normalize()

    variables_lagobs = [v for v in grupo_g2 if v.startswith("LagObs_")]
    variables_lagcal = [v for v in grupo_g2 if v.startswith("LagCal_")]
    dias_lagcal = {v: int(v.split("_")[1].replace("d", "")) for v in variables_lagcal}

    estado: dict[str, pd.DataFrame] = {}
    for producto in productos:
        sub = historial.loc[
            (historial["Nombre Producto"] == producto)
            & (historial["Fecha Venta"] < origen),
            ["Fecha Venta", "Demanda"],
        ].copy()
        estado[producto] = sub.sort_values("Fecha Venta").reset_index(drop=True)

    diario: list[dict] = []
    auditoria: list[dict] = []
    for fecha in sorted(pd.DatetimeIndex(fechas_futuras)):
        filas = []
        for producto in productos:
            serie = estado[producto]
            valores = serie.loc[serie["Fecha Venta"] < fecha, "Demanda"].tolist()
            cal = _construir_calendario([fecha]).iloc[0].to_dict()
            fila = {"Fecha Venta": fecha, "Nombre Producto": producto}
            fila.update(cal)
            for var in variables_lagobs:
                n = int(var.split("_")[1])
                fila[var] = float(valores[-n]) if len(valores) >= n else np.nan
            for var, dias in dias_lagcal.items():
                fuente = pd.Timestamp(fecha) - pd.Timedelta(days=dias)
                m = serie.loc[serie["Fecha Venta"] == fuente, "Demanda"]
                fila[var] = float(m.iloc[0]) if len(m) else np.nan
            filas.append(fila)

        filas_df = pd.DataFrame(filas)
        X_fut = filas_df[grupo_g2 + ["Nombre Producto"]]
        Xt_fut = transformar_filas_g2(preprocesador, X_fut, grupo_g2)
        pred = np.maximum(modelo.predict(Xt_fut).astype(float), 0.0)

        if auditar_estado:
            auditoria.append(
                {
                    "fecha": fecha,
                    "tamanos_estado": [len(estado[p]) for p in productos],
                    "n_predicciones": int(len(pred)),
                }
            )

        for i, producto in enumerate(productos):
            diario.append(
                {
                    "Fecha Venta": fecha,
                    "Nombre Producto": producto,
                    "Prediccion": float(pred[i]),
                }
            )
            nuevo = pd.DataFrame(
                [{"Fecha Venta": fecha, "Demanda": float(pred[i])}]
            )
            estado[producto] = pd.concat([estado[producto], nuevo], ignore_index=True)

    pronostico_diario = pd.DataFrame(diario)[COLUMNAS_DIARIO]
    pronostico_mensual = (
        pronostico_diario.groupby("Nombre Producto", sort=False)["Prediccion"]
        .sum()
        .reset_index()
        .rename(columns={"Prediccion": "Pronostico_Mensual"})
        .sort_values("Pronostico_Mensual", ascending=False)
        .reset_index(drop=True)
    )
    pronostico_mensual["Ranking"] = np.arange(1, len(pronostico_mensual) + 1)

    if auditar_estado:
        return pronostico_diario, pronostico_mensual, auditoria
    return pronostico_diario, pronostico_mensual


def auditoria_anti_leakage(bundle: dict) -> dict:
    """Comprueba que el estado se actualiza solo DESPUES de predecir
    los 10 productos de cada fecha (sin contaminacion intra-fecha)."""
    diario, mensual, auditoria = generar_pronostico_siguiente_mes(bundle, auditar_estado=True)
    fechas = construir_calendario_siguiente_mes(bundle["historial_top10"])
    base_por_producto = len(
        bundle["historial_top10"].loc[
            bundle["historial_top10"]["Fecha Venta"] < pd.Timestamp(fechas.min())
        ]
    ) // len(bundle["top10_final"])
    resultado = []
    para_cada_k = True
    for k, registro in enumerate(auditoria):
        iguales = len(set(registro["tamanos_estado"])) == 1
        esperado = base_por_producto + k
        con_k = registro["tamanos_estado"][0] == esperado
        diez = registro["n_predicciones"] == 10
        ok = iguales and con_k and diez
        resultado.append(
            {
                "fecha": registro["fecha"],
                "tamanos_estado": registro["tamanos_estado"],
                "n_predicciones": registro["n_predicciones"],
                "estado_pre_fecha_unico": iguales,
                "tamano_esperado_base_plus_k": con_k,
                "ok": ok,
            }
        )
        para_cada_k = para_cada_k and ok
    return {
        "fechas_auditadas": len(resultado),
        "base_por_producto": base_por_producto,
        "todos_ok": para_cada_k,
        "detalle": resultado,
        "pronostico_diario": diario,
        "pronostico_mensual": mensual,
    }


__all__ = [
    "PROJECT_ROOT",
    "ARTIFACT_DIR",
    "BUNDLE_PATH",
    "CONFIG_PATH",
    "MANIFEST_PATH",
    "CLAVES_BUNDLE",
    "calcular_sha256",
    "verificar_integridad_artefactos",
    "validar_bundle_y_config",
    "cargar_artefactos_produccion",
    "obtener_siguiente_mes",
    "construir_calendario_siguiente_mes",
    "transformar_filas_g2",
    "generar_pronostico_siguiente_mes",
    "auditoria_anti_leakage",
]