import streamlit as st

from utils.tensorflow_loader import load_tf_model
from utils.image_utils import preprocess_for_tf, predict_binary_tf


def run():
    st.header("🫁 Clasificador de pulmones")

    st.markdown(
        """
        Modelo para radiografías de tórax.

        Este módulo queda pendiente de validación final con el grupo:

        - clases exactas,
        - interpretación de la salida,
        - umbral final,
        - preprocesamiento esperado.
        """
    )

    uploaded_file = st.file_uploader(
        "Sube una radiografía de tórax",
        type=["jpg", "jpeg", "png"],
        key="lung_uploader"
    )

    if uploaded_file is None:
        return

    model = load_tf_model("lung")

    image_display, img_array = preprocess_for_tf(uploaded_file, model)

    st.image(image_display, caption="Imagen cargada", use_container_width=True)

    prob_positive, prob_negative = predict_binary_tf(model, img_array)

    st.subheader("Resultado provisional")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probabilidad clase positiva", f"{prob_positive:.2%}")

    with col2:
        st.metric("Probabilidad clase negativa", f"{prob_negative:.2%}")

    st.progress(prob_positive)

    st.warning(
        "Resultado provisional. Falta confirmar si la clase positiva corresponde "
        "a tumor, anomalía u otra etiqueta."
    )