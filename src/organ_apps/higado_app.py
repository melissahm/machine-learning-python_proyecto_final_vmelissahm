import streamlit as st

from utils.tensorflow_loader import load_tf_model
from utils.image_utils import (
    preprocess_liver_roi_predicted,
    predict_binary_tf,
)

SEGMENTATION_THRESHOLD = 0.30
TUMOR_THRESHOLD = 0.50
ROI_MARGIN = 10


def run():
    st.header("🟤 Análisis hepático")

    st.markdown(
        """
        Este módulo utiliza una cascada de **dos modelos de Deep Learning**:

        1. Una **U-Net** segmenta automáticamente el hígado.
        2. La región hepática aislada se envía al clasificador **V4B**
           para estimar **tumor / no tumor**.

        Solo debes subir una tomografía original tipo `volume`.
        No es necesario aportar una máscara.
        """
    )

    uploaded_file = st.file_uploader(
        "Sube una imagen de tomografía abdominal",
        type=["jpg", "jpeg", "png"],
        key="liver_uploader",
    )

    if uploaded_file is None:
        return

    segmentation_model = load_tf_model("liver_segmentation")
    tumor_model = load_tf_model("liver_tumor")

    try:
        (
            image_display,
            mask_probability,
            mask_binary,
            roi_display,
            roi_batch,
        ) = preprocess_liver_roi_predicted(
            uploaded_file=uploaded_file,
            segmentation_model=segmentation_model,
            tumor_model=tumor_model,
            segmentation_threshold=SEGMENTATION_THRESHOLD,
            margin=ROI_MARGIN,
        )

    except ValueError as error:
        st.image(
            uploaded_file,
            caption="Imagen cargada",
            use_container_width=True,
        )

        st.error("No se pudo aislar correctamente la región hepática.")
        st.info(
            f"{error} Esto puede ocurrir en cortes extremos donde "
            "el hígado aparece de forma mínima."
        )
        st.warning(
            "Modelo académico. No debe utilizarse como herramienta "
            "de diagnóstico médico."
        )
        return

    st.image(
        image_display,
        caption="Tomografía original",
        use_container_width=True,
    )

    st.subheader("Paso 1: segmentación automática del hígado")
    st.success("🟢 Se aisló una región hepática válida")

    with st.expander("Ver resultado de la segmentación", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                mask_binary * 255,
                caption="Máscara hepática predicha",
                clamp=True,
                use_container_width=True,
            )

        with col2:
            st.image(
                roi_display,
                caption="ROI enviada al modelo V4B",
                use_container_width=True,
            )

        st.caption(
            f"Umbral de segmentación: {SEGMENTATION_THRESHOLD:.2f} · "
            f"Margen de ROI: {ROI_MARGIN} píxeles"
        )

    st.divider()
    st.subheader("Paso 2: clasificación tumor / no tumor")

    prob_tumor, prob_no_tumor = predict_binary_tf(
        tumor_model,
        roi_batch,
    )

    if prob_tumor >= TUMOR_THRESHOLD:
        st.error("🔴 Posible tumor hepático detectado")
    else:
        st.success("🟢 No se detecta tumor hepático")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probabilidad de tumor", f"{prob_tumor:.2%}")

    with col2:
        st.metric("Probabilidad de no tumor", f"{prob_no_tumor:.2%}")

    st.progress(min(max(prob_tumor, 0.0), 1.0))
    st.caption(f"Umbral de clasificación: {TUMOR_THRESHOLD:.2f}")

    st.warning(
        "Modelo académico. No debe utilizarse como herramienta "
        "de diagnóstico médico."
    )