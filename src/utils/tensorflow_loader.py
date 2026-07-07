import os
import gdown
import streamlit as st
import tensorflow as tf

MODEL_DIR = "model_cache"
os.makedirs(MODEL_DIR, exist_ok=True)

TF_MODELS = {
    "brain": {
        "file": "brain_binary.keras",
        "drive_id": "1aFdiRozDfrfzWr6tMcTmZHw73dG8_qzM",
    },
    "liver_detector": {
        "file": "liver_detector.keras",
        "drive_id": "1RrBf8rwIav2cOcJBG16YKwStge_jB9jc",
    },
    "liver_tumor": {
        "file": "liver_tumor.keras",
        "drive_id": "1rETd4jIPIXxBJtIRXElP8b7_mf7ESGnB",
    },
    "lung": {
        "file": "lung_model.keras",
        "drive_id": "1I_pAnz-DncvKEhrHChHu0go4PVbd8hft",
    },
}


@st.cache_resource
def load_tf_model(model_key):
    info = TF_MODELS[model_key]
    model_path = os.path.join(MODEL_DIR, info["file"])

    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={info['drive_id']}"

        with st.spinner(f"Descargando modelo {model_key}..."):
            gdown.download(url, model_path, quiet=False)

    return tf.keras.models.load_model(model_path)