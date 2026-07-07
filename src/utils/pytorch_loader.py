import os
import gdown
import streamlit as st
import torch

MODEL_DIR = "model_cache"
os.makedirs(MODEL_DIR, exist_ok=True)

TORCH_MODELS = {
    "retina_1": {
        "file": "retina_model_1.pt",
        "drive_id": "1Ux-KE4tCPCiKiKRHqiH5DsF5Y9xC2k2W",
    },
    "retina_2": {
        "file": "retina_model_2.pt",
        "drive_id": "1bpv5lf-W7_iMifkGZqSkzd5n9x_h4WLX",
    },
}


def download_torch_model(model_key):
    info = TORCH_MODELS[model_key]
    model_path = os.path.join(MODEL_DIR, info["file"])

    if not os.path.exists(model_path):
        url = f"https://drive.google.com/uc?id={info['drive_id']}"

        with st.spinner(f"Descargando modelo {model_key}..."):
            gdown.download(url, model_path, quiet=False)

    return model_path


@st.cache_resource
def load_torch_model(model_key):
    model_path = download_torch_model(model_key)

    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()

    return model