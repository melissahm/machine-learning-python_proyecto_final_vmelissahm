import streamlit as st

from utils.tensorflow_loader import load_tf_model
from utils.image_utils import preprocess_for_tf, predict_binary_tf


def run():
    st.header("🧠 Clasificador de tumor cerebral")

    st.markdown(
        """
        Este modelo binario clasifica resonancias magnéticas cerebrales en:

        - **Tumor**
        - **No tumor**

        El entrenamiento incluyó principalmente imágenes de:

        - **Glioma**
        - **Meningioma**
        - **Tumor pituitario**

        El modelo no identifica el tipo exacto de tumor, solo estima si la imagen
        presenta características compatibles con presencia de tumor.
        """
    )

    uploaded_file = st.file_uploader(
        "Sube una resonancia magnética cerebral",
        type=["jpg", "jpeg", "png"],
        key="brain_uploader"
    )

    if uploaded_file is None:
        return

    model = load_tf_model("brain")

    image_display, img_array = preprocess_for_tf(uploaded_file, model, "brain")

    st.image(image_display, caption="Imagen cargada", use_container_width=True)

    prob_tumor, prob_notumor = predict_binary_tf(model, img_array)

    st.subheader("Resultado")

    if prob_tumor >= 0.5:
        st.error("🔴 Tumor detectado")
    else:
        st.success("🟢 No se detecta tumor")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probabilidad de tumor", f"{prob_tumor:.2%}")

    with col2:
        st.metric("Probabilidad de no tumor", f"{prob_notumor:.2%}")

    st.progress(prob_tumor)

    st.warning(
        "Modelo académico. No debe utilizarse como herramienta de diagnóstico médico."
    )