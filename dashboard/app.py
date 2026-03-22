import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# ── Configuración de la página ───────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS Commerce Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("NEXUS Commerce Intelligence")
st.markdown("Dashboard de análisis de 1,000 libros scrapeados de Books to Scrape")

# ── Cargar datos ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("../data/processed/books_clean.csv")

df = load_data()

# ── Métricas principales ─────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total libros", f"{len(df):,}")
col2.metric("Precio promedio", f"£{df['price'].mean():.2f}")
col3.metric("Precio máximo", f"£{df['price'].max():.2f}")
col4.metric("Precio mínimo", f"£{df['price'].min():.2f}")

st.divider()

# ── Gráficos ─────────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribución de ratings")
    rating_counts = df["rating"].value_counts().sort_index()
    fig1 = px.bar(
        x=rating_counts.index,
        y=rating_counts.values,
        labels={"x": "Rating (estrellas)", "y": "Cantidad de libros"},
        color=rating_counts.values,
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.subheader("Distribución de precios")
    fig2 = px.histogram(
        df, x="price", nbins=30,
        labels={"price": "Precio (£)", "count": "Cantidad"},
        color_discrete_sequence=["#534AB7"]
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Precio promedio por rating ────────────────────────────────────────────────
st.subheader("Precio promedio por rating")
avg_price = df.groupby("rating")["price"].mean().reset_index()
fig3 = px.line(
    avg_price, x="rating", y="price",
    markers=True,
    labels={"rating": "Rating (estrellas)", "price": "Precio promedio (£)"},
    color_discrete_sequence=["#0F6E56"]
)
st.plotly_chart(fig3, use_container_width=True)

# ── Predictor de rating ───────────────────────────────────────────────────────
st.divider()
st.subheader("Predictor de rating")
st.markdown("Ingresa un precio y el modelo predice el rating del libro")

price_input = st.slider("Precio del libro (£)", min_value=10.0, max_value=60.0, value=30.0, step=0.5)

model = joblib.load("../models/rating_predictor.pkl")
input_df = pd.DataFrame([[price_input]], columns=["price"])
prediction = model.predict(input_df)[0]
proba = model.predict_proba(input_df)[0]
stars = "⭐" * prediction

st.metric("Rating predicho", f"{prediction} estrellas {stars}")
st.caption(f"Confianza del modelo: {max(proba)*100:.1f}% — Accuracy general: 19.5% (baseline)")