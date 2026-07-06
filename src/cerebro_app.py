import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# CONFIGURACIÓN
# ==========================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

MODEL_PATH = "model_1_binary_tumor_v2.keras"
FILE_ID = "1aFdiRozDfrfzWr6tMcTmZHw73dG8_qzM"

IMG_SIZE = (512, 512)

# ==========================
# CARGAR MODELO
# ==========================

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        url = f"https://drive.google.com/uc?id={FILE_ID}"

        with st.spinner("Descargando modelo desde Google Drive..."):
            gdown.download(url, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ==========================
# INTERFAZ
# ==========================

st.title("🧠 Brain Tumor Detection")

st.write(
    """
    Esta aplicación utiliza una Red Neuronal Convolucional (CNN)
    entrenada para clasificar imágenes de resonancia magnética
    cerebral en dos categorías:

    - Tumor
    - No Tumor
    """
)

uploaded_file = st.file_uploader(
    "Selecciona una imagen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(
        image,
        caption="Imagen cargada",
        use_container_width=True
    )

    image = image.resize(IMG_SIZE)

    img = np.array(image)

    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0][0]

    prob_tumor = float(prediction)
    prob_notumor = 1 - prob_tumor

    st.divider()

    st.subheader("Resultado")

    if prob_tumor >= 0.5:
        st.error("🔴 Tumor detectado")
    else:
        st.success("🟢 No se detecta tumor")

    st.metric(
        "Probabilidad de Tumor",
        f"{prob_tumor:.2%}"
    )

    st.metric(
        "Probabilidad de No Tumor",
        f"{prob_notumor:.2%}"
    )

    st.progress(prob_tumor)

    st.caption(
        "⚠️ Este modelo tiene fines exclusivamente académicos y no debe utilizarse como herramienta de diagnóstico médico."
    )