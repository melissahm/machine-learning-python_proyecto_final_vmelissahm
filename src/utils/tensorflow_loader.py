import os

import gdown
import streamlit as st
import tensorflow as tf

MODEL_DIR = "model_cache"
os.makedirs(MODEL_DIR, exist_ok=True)

TF_MODELS = {
    "brain": {
        "file": "brain_binary.keras",
        "drive_id": "1HmwMc7jZuMEf3GY-k_yONYL3xZqRpV3S",
    },
    "liver_detector": {
        "file": "liver_detector.keras",
        "drive_id": "1RrBf8rwIav2cOcJBG16YKwStge_jB9jc",
    },
    "liver_segmentation": {
        "file": "model_liver_segmentation_v1.keras",
        "drive_id": "1w9E8ZKd6XVCu3v0HS5pnpp2wIYHPQlbN",
    },
    "liver_tumor": {
        "file": "model_tumor_v4b.keras",
        "drive_id": "1rETd4jIPIXxBJtIRXElP8b7_mf7ESGnB",
    },
    "lung": {
        "file": "lung_model.keras",
        "drive_id": "1I_pAnz-DncvKEhrHChHu0go4PVbd8hft",
    },
}


@st.cache_resource
def load_tf_model(model_key):
    """Descarga y carga un modelo Keras únicamente para inferencia."""

    if model_key not in TF_MODELS:
        raise KeyError(f"Modelo no configurado: {model_key}")

    info = TF_MODELS[model_key]
    model_path = os.path.join(MODEL_DIR, info["file"])

    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={info['drive_id']}"

        with st.spinner(f"Descargando modelo {model_key}..."):
            downloaded_path = gdown.download(url, model_path, quiet=False)

        if downloaded_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se pudo descargar el modelo {model_key}."
            )

    # compile=False evita tener que registrar las métricas y pérdidas
    # personalizadas usadas al entrenar la U-Net.
    return tf.keras.models.load_model(model_path, compile=False)