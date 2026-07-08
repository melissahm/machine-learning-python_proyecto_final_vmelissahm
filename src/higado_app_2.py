import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# CONFIGURACIÓN MODELOS
# ==========================

LIVER_DETECTOR_PATH = "model_liver.keras"
LIVER_DETECTOR_FILE_ID = "1RrBf8rwIav2cOcJBG16YKwStge_jB9jc"

LIVER_TUMOR_PATH = "model_tumor_v4b.keras"
LIVER_TUMOR_FILE_ID = "1rETd4jIPIXxBJtIRXElP8b7_mf7ESGnB"

# Modelo 1: detector de hígado
# Entrada esperada: 256x256x1
# Normalización: externa, dividir entre 255

LIVER_DETECTOR_SIZE = (256, 256)

# Modelo 2: tumor / no tumor
# Entrada esperada: 384x384x1
# Normalización: externa, dividir entre 255

LIVER_TUMOR_SIZE = (384, 384)

# ==========================
# CARGA DE MODELOS
# ==========================

@st.cache_resource
def load_model_from_drive(model_path, file_id):
    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        with st.spinner(f"Descargando modelo {model_path}..."):
            gdown.download(url, model_path, quiet=False)

    return tf.keras.models.load_model(model_path)


# ==========================
# PREPROCESAMIENTO
# ==========================

def preprocess_liver_image(uploaded_file, img_size):
    """
    Preprocesamiento utilizado para los modelos de hígado.

    Pasos:
    1. Leer imagen.
    2. Convertir a escala de grises.
    3. Redimensionar al tamaño requerido por el modelo.
    4. Convertir a array NumPy.
    5. Normalizar dividiendo entre 255.
    6. Añadir canal: (alto, ancho, 1).
    7. Añadir batch: (1, alto, ancho, 1).
    """

    image = Image.open(uploaded_file).convert("L")
    image_display = image.copy()

    image = image.resize(img_size)

    img_array = np.array(image).astype("float32") / 255.0

    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    return image_display, img_array


def predict_binary(model, img_array):
    prediction = model.predict(img_array, verbose=0)[0][0]

    prob_positive = float(prediction)
    prob_negative = 1 - prob_positive

    return prob_positive, prob_negative


# ==========================
# STREAMLIT APP
# ==========================

st.title("🟤 Clasificador de hígado")

st.markdown(
    """
    Esta aplicación utiliza un flujo en dos etapas:

    1. **Modelo detector de hígado**  
       Determina si la imagen contiene hígado.

    2. **Modelo tumor / no tumor**  
       Solo se ejecuta si el primer modelo detecta hígado.
    """
)

uploaded_file = st.file_uploader(
    "Sube una imagen de tomografía abdominal",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # ==========================
    # PASO 1: DETECTAR HÍGADO
    # ==========================

    liver_detector_model = load_model_from_drive(
        LIVER_DETECTOR_PATH,
        LIVER_DETECTOR_FILE_ID
    )

    image_display, img_detector = preprocess_liver_image(
        uploaded_file,
        LIVER_DETECTOR_SIZE
    )

    st.image(
        image_display,
        caption="Imagen cargada",
        use_container_width=True
    )

    prob_liver, prob_no_liver = predict_binary(
        liver_detector_model,
        img_detector
    )

    st.subheader("Paso 1: detección de hígado")

    if prob_liver >= 0.5:
        st.success("🟢 Se detecta hígado en la imagen")
    else:
        st.error("🔴 No se detecta hígado en la imagen")

    st.metric("Probabilidad de hígado", f"{prob_liver:.2%}")
    st.metric("Probabilidad de no hígado", f"{prob_no_liver:.2%}")
    st.progress(prob_liver)

    # Si no hay hígado, termina el flujo
    if prob_liver < 0.5:
        st.info(
            "Como no se detectó hígado, no se ejecuta el modelo tumor/no tumor."
        )
    else:

        # ==========================
        # PASO 2: TUMOR / NO TUMOR
        # ==========================

        st.divider()

        liver_tumor_model = load_model_from_drive(
            LIVER_TUMOR_PATH,
            LIVER_TUMOR_FILE_ID
        )

        _, img_tumor = preprocess_liver_image(
            uploaded_file,
            LIVER_TUMOR_SIZE
        )

        prob_tumor, prob_notumor = predict_binary(
            liver_tumor_model,
            img_tumor
        )

        st.subheader("Paso 2: clasificación tumor / no tumor")

        if prob_tumor >= 0.5:
            st.error("🔴 Posible tumor hepático detectado")
        else:
            st.success("🟢 No se detecta tumor hepático")

        st.metric("Probabilidad de tumor", f"{prob_tumor:.2%}")
        st.metric("Probabilidad de no tumor", f"{prob_notumor:.2%}")
        st.progress(prob_tumor)

st.caption(
    "⚠️ Modelo académico. No debe utilizarse como herramienta de diagnóstico médico."
)