import os
from pathlib import Path

import gdown
import streamlit as st

from utils.retina_preprocessing import (
    load_v06f1_model,
    load_v07_model,
)

RETINA_DIR = Path("model_cache/diabetic_retinopathy")
RETINA_DIR.mkdir(parents=True, exist_ok=True)

RETINA_MODELS = {
    "v07": {
        "file": "retina_v07_v06c_focal_soft_epoch30_screening.pth",
        "drive_id": "1Ux-KE4tCPCiKiKRHqiH5DsF5Y9xC2k2W",
    },
    "v06f1": {
        "file": "best_mini_resnet_v06_f1_retina.pth",
        "drive_id": "1bpv5lf-W7_iMifkGZqSkzd5n9x_h4WLX",
    },
}


def download_retina_model(model_key):
    info = RETINA_MODELS[model_key]
    model_path = RETINA_DIR / info["file"]

    if not model_path.exists():
        url = f"https://drive.google.com/uc?id={info['drive_id']}"
        with st.spinner(f"Descargando modelo retina {model_key}..."):
            gdown.download(url, str(model_path), quiet=False)

    return model_path


@st.cache_resource
def load_retina_models():
    download_retina_model("v06f1")
    download_retina_model("v07")

    v06f1_model = load_v06f1_model(checkpoint_dir=RETINA_DIR)
    v07_model = load_v07_model(checkpoint_dir=RETINA_DIR)

    return v06f1_model, v07_model