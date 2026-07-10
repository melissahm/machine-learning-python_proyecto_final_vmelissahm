import streamlit as st

from utils.tensorflow_loader import load_tf_model
from utils.image_utils import preprocess_for_tf


THRESHOLD = 0.5


def run():
    st.header("🫁 Clasificador de radiografías de tórax")

    st.markdown(
        """
        Este modelo binario analiza radiografías de tórax y clasifica la imagen en:

        - **Normal**
        - **Neumonía**

        **Preprocesamiento:**

        - Conversión a RGB.
        - Redimensionamiento a **128 × 128**.
        - Normalización externa dividiendo los píxeles entre 255.
        - Tensor final: **(1, 128, 128, 3)**.
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
    # ==========================
# PREPROCESAMIENTO
# ==========================

    image_display, img_array = preprocess_for_tf(
        uploaded_file,
        model,
        "lung"
    )

    st.image(
        image_display,
        caption="Radiografía cargada",
        use_container_width=True
    )

    raw_prob = float(model.predict(img_array, verbose=0)[0][0])

    prob_pneumonia = raw_prob
    prob_normal = 1 - raw_prob

    st.subheader("Resultado")

    if raw_prob >= THRESHOLD:
        st.error("🔴 Posible neumonía detectada")
    else:
        st.success("🟢 Radiografía clasificada como normal")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Probabilidad de neumonía",
            f"{prob_pneumonia:.2%}"
        )

    with col2:
        st.metric(
            "Probabilidad normal",
            f"{prob_normal:.2%}"
        )

    st.progress(prob_pneumonia)

    st.warning(
        "Modelo académico. No debe utilizarse como herramienta de diagnóstico médico."
    )


if __name__ == "__main__":
    run()