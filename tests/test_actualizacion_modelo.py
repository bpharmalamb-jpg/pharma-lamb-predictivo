# -*- coding: utf-8 -*-
"""Pruebas del flujo operacional de actualizacion del modelo.

Estas pruebas no escriben en artifacts/, artifacts_investigacion/,
artifacts_operacion/ ni en Datasources/. Los artefactos operacionales se
redirigen a tmp_path mediante monkeypatch.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import actualizacion_modelo as am
from src.forecasting import (
    ARTIFACT_DIR,
    cargar_artefactos_produccion,
    construir_calendario_siguiente_mes,
    generar_pronostico_siguiente_mes,
)


EXCEL_ORIGINAL = PROJECT_ROOT / "Datasources" / "HV_pharmalab" / "HV-(2024-2025).xlsx"
COLUMNAS_ORIGINALES = [
    "Top",
    "Comprobante Venta",
    "Fecha Venta",
    "Nombre Producto",
    "Presentacion",
    "Cantidad",
    "Categoria",
]


class DummyRandomForest:
    """Estimador liviano con la misma superficie usada por el pipeline."""

    instances = []

    def __init__(self, **params):
        self.params = params
        for key, value in params.items():
            setattr(self, key, value)
        self.estimators_ = [object()] * int(params.get("n_estimators", 1))
        DummyRandomForest.instances.append(self)

    def fit(self, X, y):
        self.fit_shape_ = tuple(X.shape)
        self.target_sum_ = float(np.sum(y))
        return self

    def predict(self, X):
        return np.full(X.shape[0], 1.0, dtype=float)


class FailingRandomForest(DummyRandomForest):
    def fit(self, X, y):
        raise RuntimeError("fallo controlado de entrenamiento")


@pytest.fixture(scope="session")
def bundle_config_investigacion():
    return cargar_artefactos_produccion(ARTIFACT_DIR)[:2]


@pytest.fixture()
def excel_base_df():
    return pd.read_excel(EXCEL_ORIGINAL, sheet_name="Rotacion", header=3)


def _excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Rotacion", index=False, startrow=3)
    return buffer.getvalue()


def _agregar_ventas_hasta(df: pd.DataFrame, productos: list[str], fecha_fin: str) -> pd.DataFrame:
    base = df.copy()
    ultima = pd.Timestamp(base["Fecha Venta"].max())
    fechas = pd.date_range(ultima + pd.Timedelta(days=1), pd.Timestamp(fecha_fin), freq="D")
    fechas = fechas[fechas.dayofweek != 6]
    registros = []
    comprobante_base = 900000
    for i, fecha in enumerate(fechas):
        for j, producto in enumerate(productos):
            registros.append(
                {
                    "Top": j + 1,
                    "Comprobante Venta": f"TEST-{comprobante_base + i:06d}-{j:02d}",
                    "Fecha Venta": fecha.strftime("%d/%m/%Y"),
                    "Nombre Producto": producto,
                    "Presentacion": "TEST",
                    "Cantidad": float((j % 3) + 1),
                    "Categoria": "TEST",
                }
            )
    return pd.concat([base, pd.DataFrame(registros)], ignore_index=True)[COLUMNAS_ORIGINALES]


def _aislar_operacion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    operacion = tmp_path / "artifacts_operacion"
    versiones = operacion / "versiones"
    monkeypatch.setattr(am, "OPERACION_DIR", operacion)
    monkeypatch.setattr(am, "VERSIONES_DIR", versiones)
    return operacion, versiones


def test_caso_1_original_hasta_diciembre_genera_enero_2026(bundle_config_investigacion):
    bundle, _ = bundle_config_investigacion
    calendario = construir_calendario_siguiente_mes(bundle["historial_top10"])
    diario, mensual = generar_pronostico_siguiente_mes(bundle)

    assert bundle["historial_top10"]["Fecha Venta"].max() == pd.Timestamp("2025-12-31")
    assert calendario.min() == pd.Timestamp("2026-01-01")
    assert calendario.max() == pd.Timestamp("2026-01-31")
    assert int((calendario.dayofweek == 6).sum()) == 0
    assert diario.shape == (270, 3)
    assert mensual.shape[0] == 10


def test_caso_2_excel_hasta_enero_genera_febrero_2026(
    monkeypatch, tmp_path, excel_base_df, bundle_config_investigacion
):
    bundle, config = bundle_config_investigacion
    _aislar_operacion(monkeypatch, tmp_path)
    DummyRandomForest.instances.clear()
    monkeypatch.setattr(am, "RandomForestRegressor", DummyRandomForest)

    df = _agregar_ventas_hasta(excel_base_df, bundle["top10_final"], "2026-01-31")
    resultado = am.reentrenar_y_guardar(_excel_bytes(df), bundle_referencia=bundle, config_referencia=config)

    assert resultado["panel"]["Fecha Venta"].max() == pd.Timestamp("2026-01-31")
    assert resultado["proximo_mes_anio"] == (2026, 2)
    assert resultado["pronostico_diario"]["Fecha Venta"].min() == pd.Timestamp("2026-02-02")
    assert resultado["pronostico_diario"]["Fecha Venta"].max() == pd.Timestamp("2026-02-28")
    assert DummyRandomForest.instances[-1].params == config["model_params"]


def test_caso_3_excel_hasta_febrero_genera_marzo_2026(
    monkeypatch, tmp_path, excel_base_df, bundle_config_investigacion
):
    bundle, config = bundle_config_investigacion
    _aislar_operacion(monkeypatch, tmp_path)
    monkeypatch.setattr(am, "RandomForestRegressor", DummyRandomForest)

    df = _agregar_ventas_hasta(excel_base_df, bundle["top10_final"], "2026-02-28")
    resultado = am.reentrenar_y_guardar(_excel_bytes(df), bundle_referencia=bundle, config_referencia=config)

    assert resultado["panel"]["Fecha Venta"].max() == pd.Timestamp("2026-02-28")
    assert resultado["proximo_mes_anio"] == (2026, 3)
    assert resultado["pronostico_diario"]["Fecha Venta"].min() == pd.Timestamp("2026-03-02")
    assert resultado["pronostico_diario"]["Fecha Venta"].max() == pd.Timestamp("2026-03-31")


def test_caso_4_columnas_incorrectas_se_rechaza(bundle_config_investigacion):
    bundle, _ = bundle_config_investigacion
    df = pd.DataFrame({"Fecha": ["01/01/2026"], "Producto": ["X"], "Unidades": [1]})

    with pytest.raises(ValueError, match="columnas requeridas"):
        am.validar_excel(_excel_bytes(df), bundle)


def test_caso_5_fecha_invalida_se_rechaza(excel_base_df, bundle_config_investigacion):
    bundle, _ = bundle_config_investigacion
    df = excel_base_df.copy()
    df.loc[0, "Fecha Venta"] = "fecha-invalida"

    resultado = am.validar_excel(_excel_bytes(df), bundle)

    assert resultado["valido"] is False
    assert resultado["errores"]
    assert any("fechas" in error.lower() for error in resultado["errores"])


def test_caso_6_sin_fecha_nueva_no_reentrena(
    monkeypatch, tmp_path, excel_base_df, bundle_config_investigacion
):
    bundle, config = bundle_config_investigacion
    operacion, _ = _aislar_operacion(monkeypatch, tmp_path)
    monkeypatch.setattr(am, "RandomForestRegressor", DummyRandomForest)

    resultado = am.validar_excel(_excel_bytes(excel_base_df), bundle)
    assert resultado["valido"] is False
    assert any("no supera" in error.lower() for error in resultado["errores"])

    with pytest.raises(ValueError, match="no supera"):
        am.reentrenar_y_guardar(_excel_bytes(excel_base_df), bundle_referencia=bundle, config_referencia=config)
    assert not (operacion / "bundle_produccion.joblib").exists()


def test_caso_7_error_entrenamiento_no_destruye_modelo_anterior(
    monkeypatch, tmp_path, excel_base_df, bundle_config_investigacion
):
    bundle, config = bundle_config_investigacion
    operacion, _ = _aislar_operacion(monkeypatch, tmp_path)
    operacion.mkdir(parents=True)
    for nombre in ["bundle_produccion.joblib", "config_modelo.json", "manifest.json"]:
        (operacion / nombre).write_bytes((ARTIFACT_DIR / nombre).read_bytes())
    hash_antes = hashlib.sha256((operacion / "bundle_produccion.joblib").read_bytes()).hexdigest()
    monkeypatch.setattr(am, "RandomForestRegressor", FailingRandomForest)

    df = _agregar_ventas_hasta(excel_base_df, bundle["top10_final"], "2026-01-31")
    with pytest.raises(RuntimeError, match="fallo controlado"):
        am.reentrenar_y_guardar(_excel_bytes(df), bundle_referencia=bundle, config_referencia=config)

    hash_despues = hashlib.sha256((operacion / "bundle_produccion.joblib").read_bytes()).hexdigest()
    assert hash_despues == hash_antes


def test_caso_8_actualizacion_correcta_queda_activa_y_muestra_nuevo_periodo(
    monkeypatch, tmp_path, excel_base_df, bundle_config_investigacion
):
    bundle, config = bundle_config_investigacion
    operacion, _ = _aislar_operacion(monkeypatch, tmp_path)
    monkeypatch.setattr(am, "RandomForestRegressor", DummyRandomForest)

    df = _agregar_ventas_hasta(excel_base_df, bundle["top10_final"], "2026-01-31")
    am.reentrenar_y_guardar(_excel_bytes(df), bundle_referencia=bundle, config_referencia=config)

    assert am.directorio_operativo_activo() == operacion
    bundle_activo, config_activa, _ = cargar_artefactos_produccion(operacion)
    calendario = construir_calendario_siguiente_mes(bundle_activo["historial_top10"])
    assert config_activa["training"]["end"] == "2026-01-31"
    assert calendario.min() == pd.Timestamp("2026-02-02")
    assert calendario.max() == pd.Timestamp("2026-02-28")
    assert int((calendario.dayofweek == 6).sum()) == 0
