import numpy as np
from PIL import Image


MODEL_PREPROCESSING = {
    "brain": {
        "normalize": False
    },
    "liver_detector": {
        "normalize": True
    },
    "liver_tumor": {
        "normalize": True
    },
    "lung": {
        "normalize": False  # pendiente confirmar
    },
}


def get_tf_input_config(model):
    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    return height, width, channels


def preprocess_for_tf(uploaded_file, model, model_key):
    height, width, channels = get_tf_input_config(model)

    if channels == 1:
        image = Image.open(uploaded_file).convert("L")
    else:
        image = Image.open(uploaded_file).convert("RGB")

    image_display = image.copy()
    image = image.resize((width, height))

    img_array = np.array(image).astype("float32")

    config = MODEL_PREPROCESSING.get(model_key, {"normalize": False})

    if config["normalize"]:
        img_array = img_array / 255.0

    if channels == 1:
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    return image_display, img_array


def predict_binary_tf(model, img_array):
    pred = model.predict(img_array, verbose=0)
    prob_positive = float(np.ravel(pred)[0])
    prob_negative = 1 - prob_positive

    return prob_positive, prob_negative