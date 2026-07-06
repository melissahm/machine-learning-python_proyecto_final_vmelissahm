import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

# ==========================
# Configuración
# ==========================

MODEL_PATH = "models/model_1_binary_tumor_v2.keras"
IMAGE_DIR = Path("data/test_images")
IMG_SIZE = (512, 512)

# ==========================
# Cargar modelo
# ==========================

model = tf.keras.models.load_model(MODEL_PATH)

valid_ext = [".jpg", ".jpeg", ".png"]

results = []

# ==========================
# Predicción
# ==========================

for img_path in IMAGE_DIR.iterdir():

    if img_path.suffix.lower() not in valid_ext:
        continue

    image = Image.open(img_path).convert("L")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array, verbose=0)[0][0]

    prob_tumor = float(pred)
    prob_notumor = 1 - prob_tumor

    pred_label = "tumor" if prob_tumor >= 0.5 else "notumor"

    # ==========================
    # Etiqueta real según el nombre
    # ==========================

    filename = img_path.name.lower()

    if "notumor" in filename:
        real_label = "notumor"
    else:
        real_label = "tumor"

    correct = (pred_label == real_label)

    results.append({
        "image": img_path.name,
        "real_label": real_label,
        "pred_label": pred_label,
        "prob_tumor": round(prob_tumor, 4),
        "prob_notumor": round(prob_notumor, 4),
        "correct": correct
    })

# ==========================
# Resultados
# ==========================

df = pd.DataFrame(results)

print("\nRESULTADOS IMÁGENES EXTERNAS")
print("=" * 60)
print(df)

accuracy = df["correct"].mean()

print("\nAccuracy en imágenes externas:", f"{accuracy:.2%}")

# ==========================
# Guardar CSV
# ==========================

df.to_csv("google_tumor_test_results.csv", index=False)

print("\nArchivo guardado: google_tumor_test_results.csv")