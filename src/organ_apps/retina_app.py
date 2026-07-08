import streamlit as st
from pathlib import Path

from utils.pytorch_loader import load_retina_models
from utils.retina_preprocessing import triage_from_image_cascade, FiveZoneTriageResult


def render_result(result: FiveZoneTriageResult):
    color_map = {
        1: ("🟢", st.success),
        2: ("🟡", st.warning),
        3: ("⚪", st.info),
        4: ("🟠", st.warning),
        5: ("🔴", st.error),
    }

    icon, alert_function = color_map.get(result.severity, ("⚪", st.info))

    st.subheader(f"{icon} {result.user_facing_label}")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Modelo v06-F1", f"{result.v06f1_prob:.3f}")

    with col2:
        st.metric("Modelo v07", f"{result.v07_prob:.3f}")

    st.markdown(f"**Acción recomendada:** {result.action}")

    alert_function(result.disclaimer)


def run():
    st.header("👁️ Screening de retinopatía diabética")

    st.markdown(
        """
        Este módulo analiza imágenes de **fondo de ojo** mediante dos modelos
        PyTorch.

        El resultado final no es una clasificación simple, sino una señal de
        triaje en cinco niveles de severidad.
        """
    )

    uploaded_file = st.file_uploader(
        "Sube una imagen de fondo de ojo",
        type=["jpg", "jpeg", "png"],
        key="retina_uploader"
    )

    if uploaded_file is None:
        st.info("📤 Sube una imagen para comenzar el análisis.")
        return

    st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)

    temp_path = Path("/tmp") / uploaded_file.name
    temp_path.write_bytes(uploaded_file.getvalue())

    try:
        with st.spinner("Analizando imagen de retina..."):
            v06f1_model, v07_model = load_retina_models()

            result = triage_from_image_cascade(
            str(temp_path),
            v06f1_model=v06f1_model,
            v07_model=v07_model,
            )

        render_result(result)

    except Exception as error:
        st.error(f"Error al procesar la imagen: {error}")

    finally:
        temp_path.unlink(missing_ok=True)

    st.warning(
        "Modelo académico. No debe utilizarse como herramienta de diagnóstico médico."
    )


if __name__ == "__main__":
    run()