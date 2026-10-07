# -*- coding: utf-8 -*-
"""Tablero operativo de pronostico de demanda - Botica Pharma Lamb (ETAPA 6.3).

Dos areas:
- Consultar pronostico: consume el motor src.forecasting (artefactos
  operacionales si existen; si no, artefactos de investigacion congelados).
- Actualizar datos y modelo: valida un Excel de ventas reales, reajusta el
  Random Forest RF_09 (hiperparametros congelados) y guarda artefactos
  operacionales versionados bajo artifacts_operacion/ (via
  src/actualizacion_modelo.py).

No duplica logica de ML, no entrena directamente, no escribe archivos
fuera de artifacts_operacion/ y no modifica los artefactos de investigacion.
"""

import json
import hmac
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.forecasting import (
    ARTIFACT_DIR,
    cargar_artefactos_produccion,
    construir_calendario_siguiente_mes,
    generar_pronostico_siguiente_mes,
)
from src.actualizacion_modelo import (
    directorio_operativo_activo,
    reentrenar_y_guardar,
    validar_excel,
)

st.set_page_config(
    page_title="Botica Pharma Lamb - Tablero Predictivo de Demanda",
    layout="wide",
)

NOMBRES_MESES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril",
    5: "Mayo", 6: "Junio", 7: "Julio", 8: "Agosto",
    9: "Septiembre", 10: "Octubre", 11: "Noviembre", 12: "Diciembre",
}

NARANJA = "#F97316"
NARANJA_OSCURO = "#C2410C"
BORDE_SUAVE = "#FED7AA"

st.markdown(
    f"""
    <style>
    .block-container {{
        max-width: 1180px;
        padding-top: 1.25rem;
        padding-bottom: 2rem;
    }}
    h1, h2, h3 {{
        letter-spacing: 0;
    }}
    div[data-testid="stTabs"] button {{
        font-weight: 650;
    }}
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {NARANJA_OSCURO};
        border-bottom-color: {NARANJA};
    }}
    .farmalab-title {{
        color: {NARANJA_OSCURO};
        font-size: 2.05rem;
        font-weight: 800;
        line-height: 1.08;
        margin: 0 0 0.15rem 0;
    }}
    .farmalab-subtitle {{
        color: #374151;
        font-size: 1.08rem;
        font-weight: 650;
        margin: 0 0 0.6rem 0;
    }}
    .info-line {{
        display: inline-block;
        color: #374151;
        background: #FFF7ED;
        border: 1px solid {BORDE_SUAVE};
        border-radius: 8px;
        padding: 0.48rem 0.7rem;
        margin: 0.15rem 0 0.75rem 0;
        font-size: 0.92rem;
    }}
    .model-note {{
        color: #6B7280;
        font-size: 0.86rem;
        margin: -0.15rem 0 0.7rem 0;
    }}
    .section-title {{
        color: #1F2937;
        font-size: 1.08rem;
        font-weight: 750;
        margin: 1rem 0 0.4rem 0;
    }}
    .kpi-card {{
        background: #FFFFFF;
        border: 1px solid {BORDE_SUAVE};
        border-left: 4px solid {NARANJA};
        border-radius: 8px;
        padding: 0.75rem 0.82rem;
        min-height: 108px;
        box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04);
    }}
    .kpi-label {{
        color: #6B7280;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 0.28rem;
    }}
    .kpi-value {{
        color: #111827;
        font-size: 1.32rem;
        font-weight: 800;
        line-height: 1.15;
        word-break: normal;
    }}
    .kpi-detail {{
        color: #6B7280;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.32rem;
    }}
    .planning-note {{
        border-left: 4px solid {NARANJA};
        background: #FFF7ED;
        border-radius: 8px;
        color: #374151;
        padding: 0.75rem 0.9rem;
        margin-top: 0.9rem;
        font-size: 0.92rem;
    }}
    div[data-testid="stSelectbox"] label {{
        color: #1F2937;
        font-weight: 700;
    }}
    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid {BORDE_SUAVE};
        border-radius: 8px;
        padding: 0.65rem 0.75rem;
    }}
    .login-title {{
        color: {NARANJA_OSCURO};
        font-size: 2rem;
        font-weight: 800;
        text-align: center;
        margin: 3rem 0 0.15rem 0;
    }}
    .login-subtitle {{
        color: #374151;
        font-size: 1.05rem;
        font-weight: 650;
        text-align: center;
        margin-bottom: 0.35rem;
    }}
    .login-caption {{
        color: #6B7280;
        font-size: 0.9rem;
        text-align: center;
        margin-bottom: 1rem;
    }}
    div[data-testid="stForm"] {{
        background: #FFFFFF;
        border: 1px solid {BORDE_SUAVE};
        border-radius: 8px;
        padding: 1.25rem 1.25rem 1rem 1.25rem;
        box-shadow: 0 8px 24px rgba(17, 24, 39, 0.06);
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


def _nombre_mes(numero):
    return NOMBRES_MESES[int(numero)]


def _etiqueta_periodo(historial) -> str:
    primer_dia = construir_calendario_siguiente_mes(historial).min()
    return f"{_nombre_mes(primer_dia.month)} {primer_dia.year}"


def _fecha_larga(fecha) -> str:
    return pd.Timestamp(fecha).strftime("%d/%m/%Y")


def _kpi(label: str, value: str, detail: str | None = None) -> None:
    detalle = f'<div class="kpi-detail">{detail}</div>' if detail else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {detalle}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _grafico_diario(datos: pd.DataFrame) -> alt.Chart:
    base = datos.sort_values("Fecha Venta").copy()
    return (
        alt.Chart(base)
        .mark_line(color=NARANJA, strokeWidth=3, point=alt.OverlayMarkDef(size=42, color=NARANJA))
        .encode(
            x=alt.X(
                "Fecha Venta:T",
                title="Fecha",
                axis=alt.Axis(format="%d/%m", tickCount=10, labelAngle=-35, grid=False),
            ),
            y=alt.Y("Prediccion:Q", title="Demanda estimada"),
            tooltip=[
                alt.Tooltip("Fecha Venta:T", title="Fecha", format="%d/%m/%Y"),
                alt.Tooltip("Prediccion:Q", title="Demanda estimada", format=",.2f"),
            ],
        )
        .properties(height=360)
    )


def _grafico_mensual(datos: pd.DataFrame) -> alt.Chart:
    base = datos.copy()
    base = base.rename(
        columns={
            "Nombre Producto": "Medicamento",
            "Pronostico_Mensual": "Total pronosticado del período",
        }
    )
    return (
        alt.Chart(base)
        .mark_bar(color=NARANJA, cornerRadiusEnd=3)
        .encode(
            x=alt.X("Total pronosticado del período:Q", title="Demanda estimada"),
            y=alt.Y(
                "Medicamento:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=420),
            ),
            tooltip=[
                alt.Tooltip("Medicamento:N"),
                alt.Tooltip("Total pronosticado del período:Q", format=",.2f"),
                alt.Tooltip("Ranking:Q", format="d"),
            ],
        )
        .properties(height=330)
    )


def _obtener_credenciales_configuradas() -> tuple[str, str] | None:
    """Lee credenciales desde st.secrets sin exponer valores sensibles."""
    try:
        auth = st.secrets.get("auth", {})
    except Exception:
        return None
    username = auth.get("username") if hasattr(auth, "get") else None
    password = auth.get("password") if hasattr(auth, "get") else None
    if not username or not password:
        return None
    return str(username), str(password)


def _credenciales_validas(usuario: str, contrasena: str, credenciales: tuple[str, str]) -> bool:
    usuario_configurado, contrasena_configurada = credenciales
    return hmac.compare_digest(
        usuario.strip().encode("utf-8"),
        usuario_configurado.encode("utf-8"),
    ) and hmac.compare_digest(
        contrasena.encode("utf-8"),
        contrasena_configurada.encode("utf-8"),
    )


def _procesar_login(credenciales: tuple[str, str] | None) -> None:
    usuario = str(st.session_state.get("usuario_login", ""))
    contrasena = str(st.session_state.get("contrasena_login", ""))
    if credenciales is None:
        st.session_state["autenticado"] = False
        st.session_state["login_error"] = "No se encontraron las credenciales de acceso configuradas."
    elif _credenciales_validas(usuario, contrasena, credenciales):
        st.session_state["autenticado"] = True
        st.session_state.pop("login_error", None)
    else:
        st.session_state["autenticado"] = False
        st.session_state["login_error"] = "Usuario o contraseña incorrectos."
    st.session_state["contrasena_login"] = ""


def _mostrar_login() -> None:
    columnas = st.columns([1, 1.05, 1])
    with columnas[1]:
        st.markdown('<div class="login-title">Botica Pharma Lamb</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="login-subtitle">Tablero Predictivo de Demanda</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="login-caption">Acceso al tablero predictivo de demanda</div>',
            unsafe_allow_html=True,
        )
        credenciales = _obtener_credenciales_configuradas()
        if credenciales is None:
            st.error("No se encontraron las credenciales de acceso configuradas.")
        with st.form("login_form"):
            usuario = st.text_input("Usuario", key="usuario_login")
            contrasena = st.text_input("Contraseña", type="password", key="contrasena_login")
            st.form_submit_button(
                "Ingresar",
                type="primary",
                on_click=_procesar_login,
                args=(credenciales,),
            )
        if st.session_state.get("login_error"):
            st.error(st.session_state["login_error"])
        if st.session_state.get("autenticado", False):
            st.rerun()


def _cerrar_sesion() -> None:
    for clave in ["autenticado", "usuario_login", "contrasena_login", "login_error"]:
        st.session_state.pop(clave, None)
    st.rerun()


def _directorio_activo() -> Path:
    operativo = directorio_operativo_activo()
    return operativo if operativo is not None else ARTIFACT_DIR


@st.cache_resource
def cachear_artefactos(directorio: str):
    return cargar_artefactos_produccion(Path(directorio))


@st.cache_data
def cachear_pronostico(directorio: str):
    bundle = cachear_artefactos(directorio)[0]
    return generar_pronostico_siguiente_mes(bundle)


if not st.session_state.get("autenticado", False):
    _mostrar_login()
    st.stop()

_, columna_cierre = st.columns([0.86, 0.14])
with columna_cierre:
    if st.button("Cerrar sesión", key="cerrar_sesion"):
        _cerrar_sesion()

directorio_actual = _directorio_activo()
es_operativo = directorio_actual != ARTIFACT_DIR

try:
    bundle, config, manifest = cachear_artefactos(str(directorio_actual))
    pronostico_diario, pronostico_mensual = cachear_pronostico(str(directorio_actual))
except Exception as e:
    st.error("No fue posible cargar el modelo predictivo de producción.")
    st.exception(e)
    st.stop()

historial = bundle["historial_top10"]
ultima_fecha_real = historial["Fecha Venta"].max().date().isoformat()
periodo_pronosticado = _etiqueta_periodo(historial)
fecha_inicio = pronostico_diario["Fecha Venta"].min()
n_fechas = int(pronostico_diario["Fecha Venta"].nunique())
n_productos = int(pronostico_mensual.shape[0])
total_pronosticado = float(pronostico_diario["Prediccion"].sum())

tab_consultar, tab_actualizar = st.tabs(
    ["Consultar pronóstico", "Actualizar datos y modelo"]
)

with tab_consultar:
    st.markdown('<div class="farmalab-title">Botica Pharma Lamb</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="farmalab-subtitle">Tablero Predictivo de Demanda</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-line">'
        f"Última actualización real: {_fecha_larga(ultima_fecha_real)} | "
        f"Pronóstico disponible: {periodo_pronosticado} | "
        "Modelo: Random Forest RF_09"
        "</div>",
        unsafe_allow_html=True,
    )
    if es_operativo:
        st.markdown(
            f'<div class="model-note">Modelo operacional activo con datos reales hasta '
            f'{_fecha_larga(ultima_fecha_real)}.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="model-note">Usando el modelo de investigación congelado. '
            "Puedes actualizarlo con un Excel histórico completo en la pestaña "
            "Actualizar datos y modelo.</div>",
            unsafe_allow_html=True,
        )

    productos = pronostico_mensual["Nombre Producto"].astype(str).tolist()
    producto_seleccionado = st.selectbox("Medicamento", productos, key="medicamento_consulta")

    diario_producto = pronostico_diario.loc[
        pronostico_diario["Nombre Producto"] == producto_seleccionado
    ].sort_values("Fecha Venta").copy()
    serie_producto = diario_producto["Prediccion"]
    fecha_inicio_producto = diario_producto["Fecha Venta"].min()
    fecha_fin_producto = diario_producto["Fecha Venta"].max()
    indice_pico = serie_producto.idxmax()
    pico_diario = float(diario_producto.loc[indice_pico, "Prediccion"])
    fecha_pico = diario_producto.loc[indice_pico, "Fecha Venta"]
    total_periodo_producto = float(serie_producto.sum())

    with st.container(border=True):
        st.markdown("### Pronóstico de demanda diaria")
        contexto_producto, contexto_periodo = st.columns([1.15, 1])
        with contexto_producto:
            st.caption("Medicamento seleccionado")
            st.markdown(f"**{producto_seleccionado}**")
        with contexto_periodo:
            st.caption("Período pronosticado")
            st.markdown(
                f"**{_fecha_larga(fecha_inicio_producto)} – "
                f"{_fecha_larga(fecha_fin_producto)}**"
            )
        st.caption(
            "Las estimaciones corresponden al mes calendario inmediatamente "
            "posterior a la última fecha real disponible."
        )

    st.markdown(
        f'<div class="section-title">Demanda diaria pronosticada — '
        f'{periodo_pronosticado}</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"{len(diario_producto)} fechas previstas de operación. "
        "Cada marcador representa la demanda estimada para una fecha."
    )
    st.altair_chart(_grafico_diario(diario_producto), width="stretch")

    tabla_diaria = diario_producto[["Fecha Venta", "Prediccion"]].rename(
        columns={
            "Fecha Venta": "Fecha",
            "Prediccion": "Demanda estimada",
        }
    )
    st.markdown(
        '<div class="section-title">Predicciones diarias por fecha</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        tabla_diaria,
        hide_index=True,
        width="stretch",
        height=360,
        column_config={
            "Fecha": st.column_config.DateColumn("Fecha", format="DD/MM/YYYY"),
            "Demanda estimada": st.column_config.NumberColumn(
                "Demanda estimada",
                format="%.2f",
            ),
        },
    )

    st.markdown('<div class="section-title">Resumen del período</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        _kpi("Promedio diario", f"{float(serie_producto.mean()):,.2f}")
    with p2:
        _kpi(
            "Pico diario estimado",
            f"{pico_diario:,.2f}",
            f"Fecha del pico: {_fecha_larga(fecha_pico)}",
        )
    with p3:
        _kpi("Total pronosticado del período", f"{total_periodo_producto:,.2f}")
    st.caption(
        "Suma de las estimaciones diarias del período, calculada con los valores "
        "originales sin redondear. No representa una cantidad automática de compra."
    )

    st.markdown(
        '<div class="planning-note">'
        "Los valores mostrados son estimaciones de demanda y constituyen información "
        "de apoyo para la planificación. No representan cantidades exactas o "
        "garantizadas para realizar pedidos."
        "</div>",
        unsafe_allow_html=True,
    )

    with st.expander("Comparación entre medicamentos y ranking"):
        st.caption(
            "Información agregada secundaria, calculada a partir de las "
            "predicciones diarias de cada medicamento."
        )
        st.markdown(
            '<div class="section-title">Totales pronosticados del período por '
            'medicamento</div>',
            unsafe_allow_html=True,
        )
        st.altair_chart(_grafico_mensual(pronostico_mensual), width="stretch")

        tabla_ranking = pronostico_mensual.copy()
        tabla_ranking = tabla_ranking.rename(
            columns={
                "Nombre Producto": "Medicamento",
                "Pronostico_Mensual": "Total pronosticado del período",
            }
        )
        tabla_ranking["Total pronosticado del período"] = tabla_ranking[
            "Total pronosticado del período"
        ].round(2)
        st.markdown(
            '<div class="section-title">Ranking agregado del período</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(
            tabla_ranking[
                ["Ranking", "Medicamento", "Total pronosticado del período"]
            ],
            hide_index=True,
            width="stretch",
            height=390,
        )

    with st.expander("Sobre el modelo"):
        st.write("**Modelo operativo:** Random Forest RF_09.")
        st.write("**Variables:** grupo G2.")
        st.write("**Unidad de predicción:** demanda diaria por medicamento.")
        st.write("**Criterio principal de selección:** Macro-WAPE.")
        st.write(
            "RF_09 obtuvo menor Macro-WAPE que SES bajo el protocolo de evaluación, "
            "aunque el desempeño varió entre medicamentos, períodos y métricas."
        )
        st.caption(
            f"Modelo entrenado con datos de {config['training']['start']} a "
            f"{config['training']['end']}. El horizonte corresponde al siguiente "
            "mes calendario y considera fechas de lunes a sábado."
        )
        if es_operativo:
            st.caption(
                f"Versión operacional: {config.get('operacion', {}).get('version', '')}"
            )

with tab_actualizar:
    st.title("Actualizar datos y modelo")
    st.markdown(
        "El archivo debe contener las columnas **Fecha Venta**, "
        "**Nombre Producto** y **Cantidad** (además de la estructura original: "
        "Top, Comprobante Venta, Presentacion, Categoria si están disponibles).\n\n"
        "Para uso operacional, sube siempre un **Excel con el historial completo "
        "actualizado**, no un archivo incremental mensual. Si la fecha final del "
        "archivo no supera la última fecha real del modelo activo, la actualización "
        "se bloqueará.\n\n"
        "El sistema valida el archivo, construye el panel de demanda sobre las "
        "fechas con actividad real, genera las variables G2 y reajusta el "
        "**Random Forest RF_09** con los hiperparámetros congelados de la "
        "investigación (sin tuning, sin XGBoost, sin clima y sin recalcular "
        "métricas de evaluación)."
    )

    archivo = st.file_uploader(
        "Archivo Excel de ventas reales (xlsx o xls)",
        type=["xlsx", "xls"],
        key="excel_ventas_reales",
    )

    if archivo is not None:
        try:
            validacion = validar_excel(archivo.getvalue(), bundle)
        except Exception as e:
            st.error(f"No fue posible validar el archivo: {e}")
            st.caption("No se modificó ningún archivo ni modelo.")
            validacion = None

        if validacion is not None:
            resumen = validacion["resumen"]
            st.subheader("Resumen de validación")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Filas válidas", resumen["n_filas_validas"])
            c2.metric("Fecha inicial", resumen["fecha_min"].date().isoformat() if resumen["fecha_min"] is not None else "-")
            c3.metric("Fecha final real", resumen["fecha_max"].date().isoformat() if resumen["fecha_max"] is not None else "-")
            c4.metric("Medicamentos en archivo", resumen["n_productos_archivo"])
            for error in resumen.get("errores", []):
                st.error(error)
            for advertencia in resumen["advertencias"]:
                st.warning(advertencia)

            if validacion["valido"]:
                if st.button(
                    "Actualizar historial y reentrenar modelo",
                    type="primary",
                    key="actualizar_modelo",
                ):
                    with st.spinner("Reentrenando Random Forest RF_09 con el historial actualizado..."):
                        try:
                            resultado = reentrenar_y_guardar(archivo.getvalue())
                        except Exception as e:
                            st.error(f"No fue posible actualizar el modelo: {e}")
                            st.caption(
                                "No se modificó ningún artefacto de investigación. "
                                "Revisa el resumen de validación."
                            )
                        else:
                            st.success("Modelo actualizado correctamente.")
                            st.write(
                                f"**Nueva última fecha real:** "
                                f"{resultado['panel']['Fecha Venta'].max().date().isoformat()}"
                            )
                            st.write(
                                f"**Nuevo periodo pronosticado:** "
                                f"{resultado['proximo_mes']}"
                            )
                            st.write(
                                f"**Demanda total estimada del nuevo período:** "
                                f"{resultado['total_pronostico']:,.2f}"
                            )
                            st.caption(f"Versión operacional: {resultado['version']}")
                            st.caption(
                                "El modelo anterior quedó respaldado en "
                                "artifacts_operacion/versiones/. La pestaña "
                                "'Consultar pronóstico' se refrescará con el "
                                "modelo operacional."
                            )
                            st.cache_resource.clear()
                            st.cache_data.clear()
                            st.rerun()
            else:
                st.error(
                    "El archivo no pasó la validación completa: revisa fechas, "
                    "cantidades y nombres de medicamentos."
                )

st.divider()
st.caption("Botica Pharma Lamb · Tablero predictivo de demanda")
st.caption("Modelo: RF09_G2_TOP10_2024_2025")
