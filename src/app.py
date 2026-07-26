from pathlib import Path

import streamlit as st
from PIL import Image

from organ_apps import cerebro_app
from organ_apps import higado_app


# =========================================================
# CONFIGURACIÓN DE LA PÁGINA
# =========================================================

st.set_page_config(
    page_title="MedImage Diagnosis",
    page_icon="🩺",
    layout="centered"
)


# =========================================================
# RUTAS DEL PROYECTO
# =========================================================

# Ruta donde se encuentra este archivo: proyecto/src
SRC_DIR = Path(__file__).resolve().parent

# Ruta raíz del repositorio: proyecto
PROJECT_ROOT = SRC_DIR.parent

# Carpeta de imágenes: proyecto/assets
ASSETS_DIR = PROJECT_ROOT / "assets"


# =========================================================
# CARGA DE IMÁGENES
# =========================================================

logo = Image.open(ASSETS_DIR / "logo.png")


# =========================================================
# CABECERA
# =========================================================

st.image(logo, use_container_width=True)

st.markdown(
    """
    <h2 style="text-align: center;">
        Detección de Tumores en Cerebro e Hígado
    </h2>
    """,
    unsafe_allow_html=True
)

st.write(
    """
    Aplicación desarrollada con **Deep Learning** para el análisis de imágenes
    médicas.

    El sistema integra dos módulos independientes:

    - **Cerebro:** clasificación de resonancias magnéticas (RM) en
      **tumor** o **no tumor** mediante una red neuronal convolucional (CNN).

    - **Hígado:** análisis de tomografías computarizadas (TC) mediante un
      modelo de **segmentación U-Net** para localizar el hígado, seguido de
      una CNN que clasifica la región hepática como **tumor** o **no tumor**.
    """
)

st.warning(
    "Esta aplicación tiene fines exclusivamente académicos y no debe "
    "utilizarse como herramienta de diagnóstico médico."
)


# =========================================================
# MENÚ LATERAL
# =========================================================

option = st.sidebar.selectbox(
    "Selecciona el estudio",
    [
        "🧠 Cerebro",
        "🟤 Hígado"
    ]
)


# =========================================================
# EJECUCIÓN DE LOS MÓDULOS
# =========================================================

if option == "🧠 Cerebro":
    cerebro_app.run()

elif option == "🟤 Hígado":
    higado_app.run()