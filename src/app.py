import streamlit as st

from organ_apps import cerebro_app
from organ_apps import higado_app
from organ_apps import pulmon_app
from organ_apps import retina_app


st.set_page_config(
    page_title="Medical Imaging Diagnosis",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Medical Imaging Diagnosis")

st.write(
    """
    Aplicación académica para la clasificación de imágenes médicas mediante
    modelos de Deep Learning.
    """
)

st.warning(
    "Esta aplicación tiene fines exclusivamente académicos y no debe utilizarse "
    "como herramienta de diagnóstico médico."
)

option = st.sidebar.selectbox(
    "Selecciona el estudio",
    [
        "🧠 Cerebro",
        "🟤 Hígado",
        "🫁 Pulmones",
        "👁️ Retinopatía diabética"
    ]
)

if option == "🧠 Cerebro":
    cerebro_app.run()

elif option == "🟤 Hígado":
    higado_app.run()

elif option == "🫁 Pulmones":
    pulmon_app.run()

elif option == "👁️ Retinopatía diabética":
    retina_app.run()