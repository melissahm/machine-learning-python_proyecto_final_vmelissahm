import streamlit as st

from organ_apps import cerebro_app
from organ_apps import higado_app

st.set_page_config(
    page_title="MedImage Diagnosis",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 MedImage Diagnosis")

st.write(
    """
      Aplicación desarrollada con **Deep Learning** para el análisis de imágenes médicas.

    El sistema integra dos módulos independientes:

    - 🧠 **Cerebro:** clasificación de resonancias magnéticas (RM) en **tumor** o **no tumor** mediante una red neuronal convolucional (CNN).
    - 🟤 **Hígado:** análisis de tomografías computarizadas (TC) mediante un modelo de **segmentación U-Net** para localizar el hígado, seguido de una CNN que clasifica la región hepática como **tumor** o **no tumor**.
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
        "🟤 Hígado"
    ]
)

if option == "🧠 Cerebro":
    cerebro_app.run()

elif option == "🟤 Hígado":
    higado_app.run()
