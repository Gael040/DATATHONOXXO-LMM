import streamlit as st
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import joblib
import numpy as np
import plotly.express as px
from geopy.distance import geodesic

df=pd.read_csv("binario_classification.csv",encoding="latin1")
df_ventas = pd.read_csv("Venta(in).csv")
del df['PORCENTAJE_CUMPLIMIENTO']

ventas_prom = df_ventas.groupby("TIENDA_ID")["VENTA_TOTAL"].mean().reset_index()
ventas_prom.rename(columns={"VENTA_TOTAL": "ventas_promedio"}, inplace=True)

df = df.merge(ventas_prom, on="TIENDA_ID", how="left")

# Cargar datos
st.set_page_config(page_title="Análisis de Tiendas OXXO", layout="wide")

st.title("🧭 Análisis y Predicción de Éxito para Nuevas Tiendas OXXO")

# ===================== KPI =====================
st.subheader("🔢 Métricas Clave")
col1, col2, col3 = st.columns(3)

total = len(df)
cumplen = df['EXITO'].sum()
pct_cumplen = cumplen / total * 100

col1.metric("Total de Tiendas", f"{total}")
col1.metric("Total de Tiendas exitosas", f"{cumplen}")
col2.metric("% Cumplen Meta", f"{pct_cumplen:.1f}%")

# ===================== MAPA =====================
st.subheader("📍 Mapa de Tiendas")
df['color_rgb'] = df['EXITO'].map({1: [0, 255, 0], 0: [255, 0, 0]})

zoom = st.slider("Zoom del mapa", min_value=8, max_value=16, value=8)
base_radius = 6000
radius = base_radius / (2 ** (zoom - 5))

st.pydeck_chart(pdk.Deck(
    map_style="mapbox://styles/mapbox/light-v9",
    initial_view_state=pdk.ViewState(
        latitude=df['LATITUD_NUM'].mean(),
        longitude=df['LONGITUD_NUM'].mean(),
        zoom=zoom,
        pitch=0,
    ),
    layers=[
        pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position='[LONGITUD_NUM, LATITUD_NUM]',
            get_color='color_rgb',
            get_radius=radius,
            radius_scale=1,
            pickable=True,
        )
    ],
))

# ===================== FILTROS Y GRÁFICOS =====================
st.subheader("📊 Análisis por Filtros")
region_sel = st.selectbox("Selecciona una plaza", ["Todas"] + df['PLAZA_CVE'].unique().tolist())
tipo_sel = st.selectbox("Nivel socioeconomico", ["Todas"] + df['NIVELSOCIOECONOMICO_DES'].unique().tolist())

df_filtro = df.copy()
if region_sel != "Todas":
    df_filtro = df_filtro[df_filtro['PLAZA_CVE'] == region_sel]
if tipo_sel != "Todas":
    df_filtro = df_filtro[df_filtro['NIVELSOCIOECONOMICO_DES'] == tipo_sel]

df_grouped = df_filtro.groupby('EXITO', as_index=False)['ventas_promedio'].mean()
df_grouped['EXITO'] = df_grouped['EXITO'].map({0: "No cumple", 1: "Cumple"})

# Crear gráfico interactivo
fig = px.bar(df_grouped,
             x='EXITO',
             y='ventas_promedio',
             color='EXITO',
             title="Venta Promedio por Cumplimiento de Meta",
             labels={'ventas_promedio': 'Venta Promedio', 'EXITO': 'Cumple Meta'},
             color_discrete_map={"Cumple": "green", "No cumple": "red"})

st.plotly_chart(fig, use_container_width=True)

# ===================== OXXO MÁS CERCANO =====================
st.subheader("📌 Buscar Tienda OXXO más Cercana")

input_lat = st.number_input("Latitud del punto", value=25.6866, step=0.0001, key="lat_usuario")
input_lon = st.number_input("Longitud del punto", value=-100.3161, step=0.0001, key="lon_usuario")

if st.button("Buscar tienda más cercana"):
    # Coordenada ingresada por el usuario
    punto_usuario = (input_lat, input_lon)

    # Calcular distancia a cada tienda
    df['distancia_m'] = df.apply(lambda row: geodesic(punto_usuario, (row['LATITUD_NUM'], row['LONGITUD_NUM'])).meters, axis=1)

    # Obtener la tienda más cercana
    tienda_cercana = df.loc[df['distancia_m'].idxmin()]

    st.success(f"🏪 Tienda más cercana: ID {tienda_cercana['TIENDA_ID']}")
    st.info(f"📏 Distancia: {tienda_cercana['distancia_m']:.2f} metros")

    # Crear DataFrames para el mapa
    df_usuario = pd.DataFrame([{'LATITUD_NUM': input_lat, 'LONGITUD_NUM': input_lon}])
    df_tienda = pd.DataFrame([{
        'LATITUD_NUM': tienda_cercana['LATITUD_NUM'],
        'LONGITUD_NUM': tienda_cercana['LONGITUD_NUM'],
        'TIENDA_ID': tienda_cercana['TIENDA_ID']
    }])

    # Mostrar mapa con ambos puntos
    st.pydeck_chart(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state=pdk.ViewState(
            latitude=input_lat,
            longitude=input_lon,
            zoom=13,
            pitch=0,
        ),
        layers=[
            # Punto ingresado (rojo)
            pdk.Layer(
                "ScatterplotLayer",
                data=df_usuario,
                get_position='[LONGITUD_NUM, LATITUD_NUM]',
                get_color='[255, 0, 0]',
                get_radius=50,
                pickable=True,
            ),
            # Tienda más cercana (amarillo)
            pdk.Layer(
                "ScatterplotLayer",
                data=df_tienda,
                get_position='[LONGITUD_NUM, LATITUD_NUM]',
                get_color='[260, 255, 0]',
                get_radius=50,
                pickable=True,
            )
        ],
        tooltip={"text": "Tienda"}
    ))

# ===================== PREDICCIÓN NUEVO PUNTO =====================
st.subheader("📍 Predicción para una Nueva Ubicación")
# Carga el dataframe de casos para evaluación
df_casos = pd.read_csv("df_test_final.csv", encoding='latin1')
df_casos.rename(columns={'CafeterÃ­as': 'Cafeterías'}, inplace=True)

# Selector para elegir TIENDA_ID
tienda_seleccionada = st.selectbox("Selecciona ID de Tienda", df_casos['TIENDA_ID'].unique())

# Obtén los datos de la tienda seleccionada
datos_tienda = df_casos[df_casos['TIENDA_ID'] == tienda_seleccionada].iloc[0]

# Mostramos la info de entrada para revisión del usuario
st.write("🧾 Datos de la tienda seleccionada:")
st.dataframe(datos_tienda.to_frame())

# Carga tu modelo
modelo = None
try:
    with open("modelo_entrenado.pkl", "rb") as f:
        modelo = joblib.load("modelo_entrenado.joblib")
except FileNotFoundError:
    st.warning("⚠️ Modelo de predicción no encontrado. Guarda tu modelo como 'modelo_entrenado.joblib'")

if modelo:
# Usamos la fila completa excepto columnas innecesarias
    columnas_modelo = [col for col in df_casos.columns if col not in ['EXITO']]  # Ajusta esto si necesitas conservar más/menos columnas
    input_data = datos_tienda[columnas_modelo].to_frame().T  # Convertimos a DataFrame de una fila

    # Si es necesario, aplica preprocesamiento aquí

    prob = modelo.predict_proba(input_data)[0][1]
    st.success(f"Probabilidad de éxito para la tienda {tienda_seleccionada}: {prob:.2%}")
