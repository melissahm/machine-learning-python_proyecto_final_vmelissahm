import streamlit as st

from utils.tensorflow_loader import load_tf_model
from utils.image_utils import preprocess_for_tf, predict_binary_tf


def run():
    st.header("🟤 Clasificador de hígado")

    st.markdown(
        """
        Este módulo utiliza **dos modelos binarios**:

        1. Primero detecta si la imagen contiene hígado.
        2. Si detecta hígado, ejecuta un segundo modelo para clasificar:
           **tumor / no tumor**.
        """
    )

    uploaded_file = st.file_uploader(
        "Sube una imagen de tomografía abdominal",
        type=["jpg", "jpeg", "png"],
        key="liver_uploader"
    )

    if uploaded_file is None:
        return

    detector_model = load_tf_model("liver_detector")

    image_display, img_array_detector = preprocess_for_tf(
        uploaded_file,
        detector_model,
        "liver_detector"
    )

    st.image(image_display, caption="Imagen cargada", use_container_width=True)

    prob_liver, prob_no_liver = predict_binary_tf(
        detector_model,
        img_array_detector
    )

    st.subheader("Paso 1: detección de hígado")

    liver_detected = prob_liver >= 0.5

    if liver_detected:
        st.success("🟢 Se detecta hígado en la imagen")
    else:
        st.error("🔴 No se detecta hígado en la imagen")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probabilidad de hígado", f"{prob_liver:.2%}")

    with col2:
        st.metric("Probabilidad de no hígado", f"{prob_no_liver:.2%}")

    st.progress(prob_liver)

    if not liver_detected:
        st.info(
            "Como el modelo no detectó hígado, no se ejecuta el clasificador tumor/no tumor."
        )
        return

    st.divider()

    st.subheader("Paso 2: clasificación tumor / no tumor")

    tumor_model = load_tf_model("liver_tumor")

    _, img_array_tumor = preprocess_for_tf(
        uploaded_file,
        tumor_model,
        "liver_tumor"
    )

    prob_tumor, prob_notumor = predict_binary_tf(
        tumor_model,
        img_array_tumor
    )

    if prob_tumor >= 0.5:
        st.error("🔴 Posible tumor hepático detectado")
    else:
        st.success("🟢 No se detecta tumor hepático")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probabilidad de tumor", f"{prob_tumor:.2%}")

    with col2:
        st.metric("Probabilidad de no tumor", f"{prob_notumor:.2%}")

    st.progress(prob_tumor)

    st.warning(
        "Modelo académico. No debe utilizarse como herramienta de diagnóstico médico."
    )


if __name__ == "__main__":
    run()