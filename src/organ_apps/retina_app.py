import streamlit as st


def run():
    st.header("👁️ Retinopatía diabética")

    st.markdown(
        """
        Módulo reservado para modelos de retinopatía diabética.

        Actualmente existen **dos modelos en PyTorch**, pero queda pendiente confirmar:

        - si actúan en cascada,
        - si uno es detector previo,
        - si son versiones alternativas,
        - qué clases devuelve cada modelo,
        - qué preprocesamiento necesita cada uno.
        """
    )

    uploaded_file = st.file_uploader(
        "Sube una imagen de retina",
        type=["jpg", "jpeg", "png"],
        key="retina_uploader"
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)

    st.info("Este módulo queda pendiente hasta confirmar el flujo del modelo PyTorch.")