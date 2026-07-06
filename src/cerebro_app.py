import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "models/model_1_binary_tumor_v2.keras"
IMG_SIZE = (512, 512)

st.set_page_config(
    page_title="Clasificador RM Cerebral",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("🧠 Clasificador de Tumor Cerebral")
st.write(
    "Sube una imagen de resonancia magnética cerebral. "
    "El modelo estimará si presenta tumor o no tumor."
)

uploaded_file = st.file_uploader(
    "Sube una imagen",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")

    st.image(
        image,
        caption="Imagen cargada",
        use_container_width=True,
        clamp=True
    )

    image_resized = image.resize(IMG_SIZE)

    img_array = np.array(image_resized)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]

    prob_tumor = float(prediction)
    prob_notumor = 1 - prob_tumor

    st.subheader("Resultado del modelo")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probabilidad tumor", f"{prob_tumor:.2%}")

    with col2:
        st.metric("Probabilidad no tumor", f"{prob_notumor:.2%}")

    if prob_tumor >= 0.5:
        st.error("Predicción final: TUMOR")
    else:
        st.success("Predicción final: NO TUMOR")

    st.warning(
        "Este modelo es un proyecto académico y no debe utilizarse como diagnóstico médico."
    )