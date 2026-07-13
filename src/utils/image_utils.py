from io import BytesIO

import cv2
import numpy as np
from PIL import Image

MODEL_PREPROCESSING = {
    "brain": {"normalize": False},
    "liver_detector": {"normalize": True},
    "liver_segmentation": {"normalize": True},
    "liver_tumor": {"normalize": True},
    "lung": {"normalize": True},
}


def _open_uploaded_image(uploaded_file):
    """Abre un UploadedFile, bytes, BytesIO o archivo compatible."""

    if isinstance(uploaded_file, (bytes, bytearray)):
        return Image.open(BytesIO(uploaded_file))

    if hasattr(uploaded_file, "getvalue"):
        return Image.open(BytesIO(uploaded_file.getvalue()))

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    return Image.open(uploaded_file)


def get_tf_input_config(model):
    """Obtiene alto, ancho y canales esperados por un modelo."""

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    return int(input_shape[1]), int(input_shape[2]), int(input_shape[3])


def preprocess_for_tf(uploaded_file, model, model_key):
    """Preprocesamiento genérico para los demás modelos de la app."""

    height, width, channels = get_tf_input_config(model)
    image = _open_uploaded_image(uploaded_file)

    if channels == 1:
        image = image.convert("L")
    elif channels == 3:
        image = image.convert("RGB")
    else:
        raise ValueError(
            f"Número de canales no compatible: {channels}."
        )

    image_display = image.copy()
    image = image.resize((width, height))
    img_array = np.array(image, dtype=np.float32)

    if model_key not in MODEL_PREPROCESSING:
        raise ValueError(
            f"No existe configuración de preprocesamiento para: {model_key}"
        )

    if MODEL_PREPROCESSING[model_key]["normalize"]:
        img_array = img_array / 255.0

    if channels == 1:
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    return image_display, img_array


def preprocess_liver_roi_predicted(
    uploaded_file,
    segmentation_model,
    tumor_model,
    segmentation_threshold=0.30,
    margin=10,
):
    """
    Genera automáticamente la ROI hepática usada por V4B.

    Flujo:
    imagen original -> U-Net -> máscara -> componente principal ->
    fondo negro -> recorte con margen -> resize para V4B -> /255.
    """

    image_pil = _open_uploaded_image(uploaded_file).convert("L")
    image_display = image_pil.copy()
    original = np.array(image_pil, dtype=np.uint8)

    if original.ndim != 2:
        raise ValueError(
            "La imagen no pudo convertirse correctamente a escala de grises."
        )

    original_height, original_width = original.shape

    seg_height, seg_width, seg_channels = get_tf_input_config(
        segmentation_model
    )

    if seg_channels != 1:
        raise ValueError(
            "El modelo de segmentación debe aceptar un canal."
        )

    image_seg = cv2.resize(
        original,
        (seg_width, seg_height),
        interpolation=cv2.INTER_LINEAR,
    )

    x_seg = image_seg.astype(np.float32) / 255.0
    x_seg = x_seg[None, :, :, None]

    mask_probability = segmentation_model.predict(
        x_seg,
        verbose=0,
    )[0, :, :, 0]

    mask_binary = (
        mask_probability >= segmentation_threshold
    ).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_binary,
        connectivity=8,
    )

    clean_mask = np.zeros_like(mask_binary, dtype=np.uint8)

    if num_labels > 1:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = 1 + int(np.argmax(component_areas))
        clean_mask[labels == largest_label] = 1

    if clean_mask.sum() == 0:
        raise ValueError(
            "No se pudo aislar una región hepática con suficiente confianza."
        )

    mask_binary_original = cv2.resize(
        clean_mask,
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )

    mask_binary_original = (
        mask_binary_original > 0
    ).astype(np.uint8)

    masked_image = original * mask_binary_original

    y_indices, x_indices = np.where(mask_binary_original > 0)

    y_min = max(int(y_indices.min()) - margin, 0)
    y_max = min(int(y_indices.max()) + margin + 1, original_height)
    x_min = max(int(x_indices.min()) - margin, 0)
    x_max = min(int(x_indices.max()) + margin + 1, original_width)

    roi = masked_image[y_min:y_max, x_min:x_max]

    if roi.size == 0:
        raise ValueError("La ROI hepática generada quedó vacía.")

    tumor_height, tumor_width, tumor_channels = get_tf_input_config(
        tumor_model
    )

    if tumor_channels != 1:
        raise ValueError(
            "El clasificador V4B debe aceptar un canal."
        )

    roi_resized = cv2.resize(
        roi,
        (tumor_width, tumor_height),
        interpolation=cv2.INTER_LINEAR,
    )

    roi_batch = roi_resized.astype(np.float32) / 255.0
    roi_batch = roi_batch[None, :, :, None]

    roi_display = Image.fromarray(roi_resized)

    return (
        image_display,
        mask_probability,
        mask_binary_original,
        roi_display,
        roi_batch,
    )


def predict_binary_tf(model, img_array):
    """Devuelve probabilidad positiva y negativa de un modelo binario."""

    pred = model.predict(img_array, verbose=0)
    prob_positive = float(np.ravel(pred)[0])
    prob_negative = 1.0 - prob_positive

    return prob_positive, prob_negative