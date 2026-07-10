import numpy as np
from PIL import Image


# =========================================================
# CONFIGURACIÓN DE PREPROCESAMIENTO POR MODELO
# =========================================================

MODEL_PREPROCESSING = {
    "brain": {
        # El modelo de cerebro ya incorpora Rescaling(1./255)
        "normalize": False
    },
    "liver_detector": {
        # El detector de hígado fue entrenado con valores entre 0 y 1
        "normalize": True
    },
    "liver_tumor": {
        # El clasificador de tumor hepático fue entrenado con /255
        "normalize": True
    },
    "lung": {
        # El modelo de neumonía fue entrenado con /255
        "normalize": True
    },
}


# =========================================================
# OBTENER FORMA DE ENTRADA DEL MODELO
# =========================================================

def get_tf_input_config(model):
    """
    Obtiene alto, ancho y número de canales esperados por el modelo.
    """

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    return height, width, channels


# =========================================================
# PREPROCESAMIENTO
# =========================================================

def preprocess_for_tf(uploaded_file, model, model_key):
    """
    Prepara una imagen para un modelo TensorFlow/Keras.

    Pasos:
    1. Obtiene el tamaño y canales requeridos por el modelo.
    2. Convierte la imagen a escala de grises o RGB.
    3. Redimensiona la imagen.
    4. Convierte la imagen a float32.
    5. Normaliza entre 0 y 1 cuando corresponde.
    6. Añade la dimensión de canal si es escala de grises.
    7. Añade la dimensión de batch.
    """

    height, width, channels = get_tf_input_config(model)

    if channels == 1:
        image = Image.open(uploaded_file).convert("L")
    elif channels == 3:
        image = Image.open(uploaded_file).convert("RGB")
    else:
        raise ValueError(
            f"Número de canales no compatible: {channels}. "
            "Solo se admiten modelos de 1 o 3 canales."
        )

    image_display = image.copy()

    image = image.resize((width, height))

    img_array = np.array(image, dtype=np.float32)

    if model_key not in MODEL_PREPROCESSING:
        raise ValueError(
            f"No existe configuración de preprocesamiento para: {model_key}"
        )

    config = MODEL_PREPROCESSING[model_key]

    if config["normalize"]:
        img_array = img_array / 255.0

    if channels == 1:
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    return image_display, img_array


# =========================================================
# PREDICCIÓN BINARIA
# =========================================================

def predict_binary_tf(model, img_array):
    """
    Ejecuta una predicción binaria.

    Devuelve:
    - prob_positive: probabilidad de la clase positiva.
    - prob_negative: probabilidad de la clase negativa.
    """

    pred = model.predict(img_array, verbose=0)

    prob_positive = float(np.ravel(pred)[0])
    prob_negative = 1.0 - prob_positive

    return prob_positive, prob_negative