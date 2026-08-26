# -*- coding: utf-8 -*-
"""Pruebas T1-T18 del motor de inferencia (ETAPA 6.1) + auditoria anti-leakage.

Usa solamente `src.forecasting` y los artefactos ya persistidos.
Ejecucion: python tests/test_forecasting_engine.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.forecasting import (
    ARTIFACT_DIR,
    BUNDLE_PATH,
    CLAVES_BUNDLE,
    CONFIG_PATH,
    MANIFEST_PATH,
    REFERENCIA_ENERO_2026,
    auditoria_anti_leakage,
    calcular_sha256,
    cargar_artefactos_produccion,
    construir_calendario_siguiente_mes,
    generar_pronostico_siguiente_mes,
    obtener_siguiente_mes,
    transformar_filas_g2,
    verificar_integridad_artefactos,
)

BUNDLE, CONFIG, MANIFEST = cargar_artefactos_produccion()

RESULTADOS = []


def probar(nombre, condicion):
    RESULTADOS.append((nombre, bool(condicion)))


def t1_artefactos_existen():
    probar(
        "T1 los 3 artefactos existen",
        BUNDLE_PATH.is_file() and CONFIG_PATH.is_file() and MANIFEST_PATH.is_file(),
    )


def t2_hash_bundle_coincide():
    integridad = verificar_integridad_artefactos()
    registro = MANIFEST["artifacts"]["bundle_produccion.joblib"]
    probar(
        "T2 hash bundle coincide con manifest",
        integridad["bundle_sha256"] == registro["sha256"]
        and calcular_sha256(BUNDLE_PATH) == registro["sha256"],
    )


def t3_hash_config_coincide():
    integridad = verificar_integridad_artefactos()
    registro = MANIFEST["artifacts"]["config_modelo.json"]
    probar(
        "T3 hash config coincide con manifest",
        integridad["config_sha256"] == registro["sha256"]
        and calcular_sha256(CONFIG_PATH) == registro["sha256"],
    )


def t4_schema_model_id_consistentes():
    probar(
        "T4 schema/model_id consistentes",
        BUNDLE["schema_version"] == "1.0.0"
        and CONFIG["schema_version"] == "1.0.0"
        and MANIFEST["schema_version"] == "1.0.0"
        and BUNDLE["model_id"] == "RF09_G2_TOP10_2024_2025"
        and CONFIG["model_id"] == "RF09_G2_TOP10_2024_2025"
        and MANIFEST["model_id"] == "RF09_G2_TOP10_2024_2025",
    )


def t5_bundle_ocho_claves():
    probar("T5 bundle tiene las 8 claves exactas", set(BUNDLE.keys()) == set(CLAVES_BUNDLE))


def t6_rf_09_exacto():
    rf = BUNDLE["rf_produccion"]
    ok = (
        rf.n_estimators == 500
        and rf.max_depth == 8
        and rf.min_samples_split == 2
        and rf.min_samples_leaf == 1
        and rf.max_features == "sqrt"
        and rf.random_state == 42
        and rf.n_jobs == -1
        and len(rf.estimators_) == 500
    )
    probar("T6 RF_09 exacto y 500 arboles", ok)


def t7_g2_22_exacto():
    probar(
        "T7 G2 = 22 exacto",
        len(BUNDLE["grupo_g2"]) == 22
        and len(set(BUNDLE["grupo_g2"])) == 22
        and list(BUNDLE["grupo_g2"]) == CONFIG["feature_group"]["variables"]
        and CONFIG["feature_group"]["total_variables"] == 22
    )


def t8_top10_10_exacto():
    probar(
        "T8 Top10 = 10 exacto",
        len(BUNDLE["top10_final"]) == 10
        and list(BUNDLE["top10_final"]) == CONFIG["top10"]["products"]
        and CONFIG["top10"]["count"] == 10
    )


def t9_historico_6190x3():
    hist = BUNDLE["historial_top10"]
    ok = (
        hist.shape == (6190, 3)
        and list(hist.columns) == ["Fecha Venta", "Nombre Producto", "Demanda"]
        and pd.api.types.is_datetime64_any_dtype(hist["Fecha Venta"])
        and hist["Nombre Producto"].nunique() == 10
        and hist["Fecha Venta"].max() == pd.Timestamp("2025-12-31")
    )
    probar("T9 historico = 6190x3 y max date 2025-12-31", ok)


def t10_preprocesador():
    pp = BUNDLE["preprocesador_produccion"]
    ok = (
        set(pp.keys()) == {"imputer", "encoder", "nombres"}
        and len(pp["nombres"]) == 32
        and len(pp["encoder"].categories_[0]) == 10
        and hasattr(pp["imputer"], "statistics_")
    )
    probar("T10 preprocesador imputer/encoder/nombres = 32 y 10 categorias", ok)


def t11_siguiente_mes():
    probar("T11 siguiente mes = enero 2026", obtener_siguiente_mes(BUNDLE["historial_top10"]) == (2026, 1))


def t12_calendario_enero():
    fechas = construir_calendario_siguiente_mes(BUNDLE["historial_top10"])
    domingos_enero = {4, 11, 18, 25}
    todos_los_dias = pd.DatetimeIndex(pd.date_range("2026-01-01", "2026-01-31", freq="D"))
    domingos_reales = set(todos_los_dias[todos_los_dias.dayofweek == 6].day.tolist())
    ok = (
        len(fechas) == 27
        and int((fechas.dayofweek == 6).sum()) == 0
        and domingos_reales == domingos_enero
        and fechas.min() == pd.Timestamp("2026-01-01")
        and fechas.max() == pd.Timestamp("2026-01-31")
    )
    probar("T12 calendario enero = 27 fechas, 0 domingos", ok)


def t13_diario_270x3():
    diario, _ = generar_pronostico_siguiente_mes(BUNDLE)
    ok = (
        diario.shape == (270, 3)
        and list(diario.columns) == ["Fecha Venta", "Nombre Producto", "Prediccion"]
        and diario["Nombre Producto"].nunique() == 10
        and diario["Fecha Venta"].nunique() == 27
        and len(diario.drop_duplicates(["Fecha Venta", "Nombre Producto"])) == 270
    )
    probar("T13 pronostico diario = 270x3, 10 productos, 27 fechas, sin duplicados", ok)


def t14_predicciones_finitas():
    diario, _ = generar_pronostico_siguiente_mes(BUNDLE)
    p = diario["Prediccion"].to_numpy(dtype=float)
    ok = (
        np.isfinite(p).all()
        and not (p < 0).any()
        and not np.isnan(p).any()
        and not np.isinf(p).any()
    )
    probar("T14 Prediccion sin NaN/inf/-inf ni negativos", ok)


def t15_mensual_consistente():
    diario, mensual = generar_pronostico_siguiente_mes(BUNDLE)
    ok = (
        len(mensual) == 10
        and set(mensual.columns) >= {"Nombre Producto", "Pronostico_Mensual"}
        and np.isclose(
            float(mensual["Pronostico_Mensual"].sum()),
            float(diario["Prediccion"].sum()),
            rtol=1e-9,
            atol=1e-6,
        )
    )
    probar("T15 mensual = 10 productos y suma coincide con diaria", ok)


def t16_total_referencia():
    diario, _ = generar_pronostico_siguiente_mes(BUNDLE)
    total = float(diario["Prediccion"].sum())
    probar(
        "T16 total enero coincide con 8829.307412205015",
        np.isclose(total, REFERENCIA_ENERO_2026, rtol=1e-9, atol=1e-6),
    )
    print("     total enero obtenido:", repr(total), "| referencia:", repr(REFERENCIA_ENERO_2026))


def t17_reproducible():
    d1, _ = generar_pronostico_siguiente_mes(BUNDLE)
    d2, _ = generar_pronostico_siguiente_mes(BUNDLE)
    probar(
        "T17 segunda ejecucion 270 np.allclose",
        np.allclose(d2["Prediccion"].to_numpy(), d1["Prediccion"].to_numpy()),
    )


def t18_historial_no_modificado():
    snapshot = BUNDLE["historial_top10"].copy()
    _ = generar_pronostico_siguiente_mes(BUNDLE)
    _ = generar_pronostico_siguiente_mes(BUNDLE)
    probar(
        "T18 historial del bundle no modificado",
        BUNDLE["historial_top10"].equals(snapshot)
        and np.array_equal(
            BUNDLE["historial_top10"].to_numpy(), snapshot.to_numpy()
        ),
    )


def t19_anti_leakage():
    auditoria = auditoria_anti_leakage(BUNDLE)
    probar(
        "T19 anti-leakage: estado pre-fecha unico y actualizado tras 10 predicciones",
        auditoria["todos_ok"]
        and len(auditoria["detalle"]) == 27
        and auditoria["base_por_producto"] == 619,
    )


def main():
    t1_artefactos_existen()
    t2_hash_bundle_coincide()
    t3_hash_config_coincide()
    t4_schema_model_id_consistentes()
    t5_bundle_ocho_claves()
    t6_rf_09_exacto()
    t7_g2_22_exacto()
    t8_top10_10_exacto()
    t9_historico_6190x3()
    t10_preprocesador()
    t11_siguiente_mes()
    t12_calendario_enero()
    t13_diario_270x3()
    t14_predicciones_finitas()
    t15_mensual_consistente()
    t16_total_referencia()
    t17_reproducible()
    t18_historial_no_modificado()
    t19_anti_leakage()

    fallidas = [n for n, c in RESULTADOS if not c]
    for nombre, condicion in RESULTADOS:
        print(("OK   " if condicion else "FALLO") + " " + nombre)
    print()
    if fallidas:
        print("Pruebas fallidas:", fallidas)
        raise SystemExit(1)
    print("T1-T18: 18/18 OK.")
    print("T19 (anti-leakage): OK.")
    print(
        "ETAPA 6.1 — MOTOR DE INFERENCIA INDEPENDIENTE DE NOTEBOOKS "
        "CON PRUEBAS T1-T18 APROBADAS."
    )


if __name__ == "__main__":
    main()