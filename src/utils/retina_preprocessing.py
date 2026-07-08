"""Módulo de retinopatía diabética — archivo único para distribución.

Este archivo fusiona todo el pipeline de inferencia de retina en un único
script autocontenido. Se genera a partir de la arquitectura modular del repo;
cuando se actualice algún componente, regenerar desde los archivos fuente.

Dependencias: torch, opencv-python, Pillow, numpy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# =============================================================================
# CONFIG (antes config.py)
# =============================================================================

TARGET_SIZE = 384
"""Resolución espacial de entrada usada por los tensores de 6 canales."""

DROPOUT = 0.5
"""Probabilidad de dropout usada por ``MiniResNetV06``."""

BATCH_SIZE = 16
"""Tamaño de batch por defecto para inferencia."""

CKPT_V06F1 = "best_mini_resnet_v06_f1_retina.pth"
"""Checkpoint afinado por F1 para la tarea sano-vs-enfermo (loss BCE)."""

CKPT_V07 = "retina_v07_v06c_focal_soft_epoch30_screening.pth"
"""Checkpoint final de screening entrenado con focal loss (test-pass)."""

V06F1_SAFE_HEALTHY_THRESHOLD = 0.015
"""Umbral muy bajo sobre v0.6-F1 para etiquetar un caso como sanamente confiable."""

V07_SAFE_SICK_THRESHOLD = 0.40
"""Umbral alto sobre v0.7 para etiquetar un caso como enfermamente confiable."""

V07_FINAL_THRESHOLD = 0.20
"""Umbral de decisión binaria usado para el resultado PASS final del test de v0.7."""

V06F1_PROBABLY_HEALTHY_THRESHOLD = 0.10
"""Límite superior para la zona orientativa 'probablemente sano' sobre v0.6-F1."""

V07_PROBABLY_SICK_THRESHOLD = 0.20
"""Límite inferior para la zona orientativa 'probablemente enfermo' sobre v0.7."""

CLASS_NAMES = {
    0: "No diabetic retinopathy",
    1: "Diabetic retinopathy",
}
"""Nombres legibles de las clases para la clasificación binaria."""

CANONICAL_SCALE = 300
"""Radio del ojo objetivo en píxeles usado por el pipeline canónico de Graham (2015)."""


# =============================================================================
# MODEL (antes model.py)
# =============================================================================

class ResidualBlockV06(nn.Module):
    """Bloque residual único con proyección opcional de atajo 1x1."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class MiniResNetV06(nn.Module):
    """Red residual desde cero para tensores de fondo de ojo de 6 canales y 384x384.

    Forma de entrada esperada: ``(N, 6, 384, 384)``.
    Forma de salida: ``(N, 1)`` logit en crudo.
    """

    def __init__(self, in_channels: int = 6, dropout: float = DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = ResidualBlockV06(64, 128, stride=2)
        self.layer2 = ResidualBlockV06(128, 256, stride=2)
        self.layer3 = ResidualBlockV06(256, 512, stride=2)
        self.layer4 = ResidualBlockV06(512, 512, stride=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.head(x)


def build_model(device: torch.device | None = None, dropout: float = DROPOUT) -> MiniResNetV06:
    """Crea un ``MiniResNetV06`` listo para evaluar sin cargar pesos."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniResNetV06(in_channels=6, dropout=dropout)
    model.to(device)
    model.eval()
    return model


# =============================================================================
# PREPROCESSING (antes preprocessing.py)
# =============================================================================

class PreprocessingError(ValueError):
    """Se lanza cuando una entrada no puede convertirse en un tensor de retina válido."""


InputImage = Union[str, os.PathLike, np.ndarray, Image.Image]
"""Entradas de imagen soportadas por el pipeline de preprocesamiento."""


def _validate_positive_int(value: int, name: str) -> int:
    """Valida parámetros enteros de geometría de imagen."""
    if not isinstance(value, Integral) or value <= 0:
        raise PreprocessingError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _validar_forma_imagen(arr: np.ndarray) -> np.ndarray:
    """Normaliza la forma de un array de imagen a ``(H, W, 3)`` y valida geometría."""
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim != 3:
        raise PreprocessingError(
            f"Expected a 2-D or 3-D image array, got shape {arr.shape}."
        )
    else:
        channels = arr.shape[2]
        if channels == 4:
            arr = arr[:, :, :3]
        elif channels == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif channels != 3:
            raise PreprocessingError(
                f"Expected 1, 3, or 4 channels, got {channels} channels."
            )

    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise PreprocessingError(
            f"Image height and width must be positive, got shape {arr.shape}."
        )
    return arr


def _validar_dimensiones_positivas(image: np.ndarray) -> None:
    """Verifica que una imagen tenga alto y ancho positivos."""
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        raise PreprocessingError(
            f"Image height and width must be positive, got shape {image.shape}."
        )


def load_image(input_image: InputImage) -> np.ndarray:
    """Carga y valida una imagen como un array ``uint8`` RGB de numpy."""
    if isinstance(input_image, (str, os.PathLike)):
        try:
            with Image.open(input_image) as pil_img:
                pil_img.load()
                pil_img = pil_img.convert("RGB")
                arr = np.asarray(pil_img)
        except Exception as exc:
            raise PreprocessingError(
                f"Could not open image from path '{input_image}': {exc}"
            ) from exc
    elif isinstance(input_image, Image.Image):
        arr = np.asarray(input_image.convert("RGB"))
    elif isinstance(input_image, np.ndarray):
        arr = np.asarray(input_image)
    else:
        raise PreprocessingError(
            f"Unsupported image type: {type(input_image).__name__}. "
            "Expected a path, PIL.Image, or np.ndarray."
        )

    arr = _validar_forma_imagen(arr)

    if not (
        np.issubdtype(arr.dtype, np.integer)
        or np.issubdtype(arr.dtype, np.floating)
    ):
        raise PreprocessingError(
            f"Image array must contain real numeric pixel values, got dtype {arr.dtype}."
        )

    if not np.all(np.isfinite(arr)):
        raise PreprocessingError("Image contains non-finite pixel values.")

    if arr.dtype == np.uint8:
        rgb = arr
    elif np.issubdtype(arr.dtype, np.floating):
        if arr.max() <= 1.0:
            rgb = (arr * 255.0).clip(0.0, 255.0)
        else:
            rgb = arr.clip(0.0, 255.0)
        rgb = rgb.astype(np.uint8)
    else:
        rgb = arr.clip(0, 255).astype(np.uint8)

    return rgb


def scale_radius(image: np.ndarray, target_radius: int = CANONICAL_SCALE) -> np.ndarray:
    """Normaliza el radio del ojo entre imágenes usando la heurística de la línea de escaneo central."""
    target_radius = _validate_positive_int(target_radius, "target_radius")
    _validar_dimensiones_positivas(image)
    h, w = image.shape[:2]

    middle_row = image[h // 2, :, :].sum(axis=1)
    threshold = middle_row.mean() / 10.0
    non_black_count = int((middle_row > threshold).sum())
    radius = non_black_count // 2

    if radius <= 0:
        return image

    scale = target_radius / radius
    return cv2.resize(image, (0, 0), fx=scale, fy=scale)


def ben_graham_canonical(
    image: np.ndarray, scale: int = CANONICAL_SCALE
) -> np.ndarray:
    """Aplica la mejora canónica de Ben Graham (2015)."""
    scale = _validate_positive_int(scale, "scale")
    scaled = scale_radius(image, target_radius=scale)

    sigma = scale / 30.0
    blurred = cv2.GaussianBlur(scaled, (0, 0), sigmaX=sigma)
    enhanced = cv2.addWeighted(
        scaled.astype(np.float32), 4.0,
        blurred.astype(np.float32), -4.0,
        128.0,
    )

    h, w = enhanced.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (w // 2, h // 2), int(scale * 0.9), 1.0, -1, cv2.LINE_AA)
    mask_3ch = np.stack([mask, mask, mask], axis=2)

    enhanced = enhanced * mask_3ch + 128.0 * (1.0 - mask_3ch)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def crop_to_square_centered(
    image: np.ndarray, target_size: int = TARGET_SIZE
) -> np.ndarray:
    """Recorta un cuadrado centrado y lo redimensiona a ``target_size``."""
    target_size = _validate_positive_int(target_size, "target_size")
    _validar_dimensiones_positivas(image)
    h, w = image.shape[:2]

    side = min(h, w)
    cx, cy = w // 2, h // 2
    x1 = max(cx - side // 2, 0)
    y1 = max(cy - side // 2, 0)
    x2 = min(x1 + side, w)
    y2 = min(y1 + side, h)
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (target_size, target_size))


def build_six_channel_canonical(
    image_rgb: np.ndarray, image_bg: np.ndarray
) -> torch.Tensor:
    """Apila el RGB normalizado con el Ben Graham centrado en un tensor de 6 canales."""
    rgb_norm = image_rgb.astype(np.float32) / 255.0
    bg_centered = (image_bg.astype(np.float32) - 128.0) / 128.0

    rgb_tensor = torch.from_numpy(rgb_norm.transpose(2, 0, 1)).float()
    bg_tensor = torch.from_numpy(bg_centered.transpose(2, 0, 1)).float()
    return torch.cat([rgb_tensor, bg_tensor], dim=0)


def preprocess_image(
    input_image: InputImage,
    *,
    target_size: int = TARGET_SIZE,
    canonical_scale: int = CANONICAL_SCALE,
    batched: bool = True,
) -> torch.Tensor:
    """Preprocesa una imagen de retina recién subida para inferencia con ``MiniResNetV06``."""
    target_size = _validate_positive_int(target_size, "target_size")
    canonical_scale = _validate_positive_int(canonical_scale, "canonical_scale")

    rgb = load_image(input_image)

    rgb_scaled = scale_radius(rgb, target_radius=canonical_scale)
    rgb_cropped = crop_to_square_centered(rgb_scaled, target_size)

    bg_canonical = ben_graham_canonical(rgb, scale=canonical_scale)
    bg_cropped = crop_to_square_centered(bg_canonical, target_size)

    tensor = build_six_channel_canonical(rgb_cropped, bg_cropped)

    if not torch.all(torch.isfinite(tensor)):
        raise PreprocessingError("Preprocessing produced non-finite tensor values.")

    expected_shape = (6, target_size, target_size)
    if tensor.shape != expected_shape:
        raise PreprocessingError(
            f"Unexpected output shape {tuple(tensor.shape)} (expected {expected_shape})."
        )

    return tensor.unsqueeze(0) if batched else tensor


# =============================================================================
# INFERENCE (antes inference.py)
# =============================================================================

def load_checkpoint(checkpoint_path: str | Path, device: torch.device | None = None) -> MiniResNetV06:
    """Carga un checkpoint de ``MiniResNetV06`` y lo pone en modo evaluación."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)
    model = build_model(device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =============================================================================
# TRIAGE (antes triage.py)
# =============================================================================

@dataclass(frozen=True)
class FiveZoneTriageResult:
    """Contenedor para una única decisión de triage de cinco zonas."""

    label: str
    severity: int
    reason: str
    user_facing_label: str
    action: str
    disclaimer: str
    v06f1_prob: float
    v07_prob: float
    safe_healthy_threshold: float
    probably_healthy_threshold: float
    probably_sick_threshold: float
    safe_sick_threshold: float


_FIVE_ZONE_DEFINITIONS = {
    "safe_healthy": {
        "severity": 1,
        "user_facing_label": "Señal baja del modelo",
        "action": (
            "El modelo muestra una señal baja de retinopatía diabética. Continúe con "
            "los controles de rutina recomendados por su profesional de salud visual."
        ),
        "disclaimer": (
            "Este es un prototipo educativo, no un diagnóstico clínico. Siga siempre "
            "las indicaciones de un profesional de la salud cualificado."
        ),
    },
    "probably_healthy": {
        "severity": 2,
        "user_facing_label": "Señal baja a moderada del modelo",
        "action": (
            "El resultado se inclina hacia una señal baja, pero es solo orientativo. "
            "Programe controles periódicos y consulte a un especialista si presenta síntomas."
        ),
        "disclaimer": (
            "Esta zona orientativa no es lo suficientemente fuerte como para un alta. "
            "No debe reemplazar el criterio clínico."
        ),
    },
    "uncertain": {
        "severity": 3,
        "user_facing_label": "Señal incierta del modelo",
        "action": (
            "La predicción cae en la zona gris. Por favor, haga que un oftalmólogo o "
            "especialista en retina revise esta imagen."
        ),
        "disclaimer": (
            "El modelo es menos fiable en este rango. Se requiere revisión por un "
            "especialista antes de cualquier decisión clínica."
        ),
    },
    "probably_sick": {
        "severity": 4,
        "user_facing_label": "Señal moderada del modelo",
        "action": (
            "El modelo sugiere una señal moderada de retinopatía diabética. Por favor, "
            "gestione una revisión con un especialista a la brevedad."
        ),
        "disclaimer": (
            "Esta zona orientativa no es un hallazgo positivo confiable. Confírmelo "
            "con un oftalmólogo antes de cualquier decisión de tratamiento."
        ),
    },
    "safe_sick": {
        "severity": 5,
        "user_facing_label": "Señal alta del modelo",
        "action": (
            "El modelo muestra una señal alta de retinopatía diabética. Busque "
            "evaluación por un especialista lo antes posible."
        ),
        "disclaimer": (
            "Esta es la señal más fuerte que el prototipo puede emitir, pero sigue "
            "sin ser un diagnóstico clínico. Se requiere revisión urgente por un especialista."
        ),
    },
}


def triage_five_zones(
    v06f1_prob: Union[float, np.ndarray],
    v07_prob: Union[float, np.ndarray],
    *,
    safe_healthy_threshold: float = V06F1_SAFE_HEALTHY_THRESHOLD,
    probably_healthy_threshold: float = V06F1_PROBABLY_HEALTHY_THRESHOLD,
    probably_sick_threshold: float = V07_PROBABLY_SICK_THRESHOLD,
    safe_sick_threshold: float = V07_SAFE_SICK_THRESHOLD,
) -> Union[FiveZoneTriageResult, list[FiveZoneTriageResult]]:
    """Clasifica uno o más casos de retina en cinco zonas de triage."""
    is_scalar = np.isscalar(v06f1_prob) and np.isscalar(v07_prob)

    v06 = np.atleast_1d(np.asarray(v06f1_prob, dtype=float))
    v07 = np.atleast_1d(np.asarray(v07_prob, dtype=float))

    if v06.shape != v07.shape:
        raise ValueError(
            f"v06f1_prob and v07_prob must have the same shape: {v06.shape} vs {v07.shape}"
        )

    condiciones = [
        v06 < safe_healthy_threshold,
        v07 > safe_sick_threshold,
        v06 < probably_healthy_threshold,
        v07 >= probably_sick_threshold,
    ]
    elecciones = ["safe_healthy", "safe_sick", "probably_healthy", "probably_sick"]
    labels = np.select(condiciones, elecciones, default="uncertain")

    reason_elecciones = [
        f"v06f1 probability below safe-healthy threshold ({safe_healthy_threshold})",
        f"v07 probability above safe-sick threshold ({safe_sick_threshold})",
        f"v06f1 probability below probably-healthy threshold ({probably_healthy_threshold})",
        f"v07 probability at or above probably-sick threshold ({probably_sick_threshold})",
    ]
    reasons = np.select(
        condiciones,
        reason_elecciones,
        default="probabilities fall in the uncertain gray zone between orientative thresholds",
    )

    results: list[FiveZoneTriageResult] = []
    for i in range(v06.size):
        label = str(labels.flat[i])
        meta = _FIVE_ZONE_DEFINITIONS[label]
        results.append(
            FiveZoneTriageResult(
                label=label,
                severity=meta["severity"],
                reason=str(reasons.flat[i]),
                user_facing_label=meta["user_facing_label"],
                action=meta["action"],
                disclaimer=meta["disclaimer"],
                v06f1_prob=float(v06.flat[i]),
                v07_prob=float(v07.flat[i]),
                safe_healthy_threshold=safe_healthy_threshold,
                probably_healthy_threshold=probably_healthy_threshold,
                probably_sick_threshold=probably_sick_threshold,
                safe_sick_threshold=safe_sick_threshold,
            )
        )

    if is_scalar:
        return results[0]
    return results


# =============================================================================
# SERVICE (antes service.py)
# =============================================================================

DIRECTORIO_MODELOS: Path = Path("models/diabetic_retinopathy")
"""Directorio por defecto donde se buscan los checkpoints, relativo al CWD."""


def load_v06f1_model(
    checkpoint_dir: str | Path | None = None,
    device: torch.device | None = None,
) -> MiniResNetV06:
    """Carga el modelo v0.6-F1 (BCE) usado para la zona 'sano seguro'."""
    directorio = Path(checkpoint_dir) if checkpoint_dir is not None else DIRECTORIO_MODELOS
    return load_checkpoint(directorio / CKPT_V06F1, device=device)


def load_v07_model(
    checkpoint_dir: str | Path | None = None,
    device: torch.device | None = None,
) -> MiniResNetV06:
    """Carga el modelo v0.7 (focal) usado para la zona 'enfermo seguro'."""
    directorio = Path(checkpoint_dir) if checkpoint_dir is not None else DIRECTORIO_MODELOS
    return load_checkpoint(directorio / CKPT_V07, device=device)


def predict_two_model_probabilities(
    image_input: InputImage,
    *,
    v06f1_model: MiniResNetV06,
    v07_model: MiniResNetV06,
    device: torch.device | None = None,
) -> tuple[float, float]:
    """Preprocesa una imagen y devuelve las probabilidades de ambos modelos."""
    if device is None:
        device = next(v07_model.parameters()).device

    tensor = preprocess_image(image_input, batched=True)
    tensor = tensor.to(device)

    with torch.no_grad():
        v06f1_logit = v06f1_model(tensor)
        v07_logit = v07_model(tensor)
        v06f1_prob = float(torch.sigmoid(v06f1_logit).item())
        v07_prob = float(torch.sigmoid(v07_logit).item())

    return v06f1_prob, v07_prob


def triage_from_probabilities(
    v06f1_prob: float,
    v07_prob: float,
) -> FiveZoneTriageResult:
    """Devuelve una decisión de triage de cinco zonas a partir de las probabilidades."""
    result = triage_five_zones(v06f1_prob, v07_prob)
    if isinstance(result, list):
        return result[0]
    return result


def triage_from_image(
    image_input: InputImage,
    *,
    v06f1_model: MiniResNetV06,
    v07_model: MiniResNetV06,
    device: torch.device | None = None,
) -> FiveZoneTriageResult:
    """Ejecuta el flujo completo de retina y devuelve una decisión de triage de cinco zonas."""
    v06f1_prob, v07_prob = predict_two_model_probabilities(
        image_input,
        v06f1_model=v06f1_model,
        v07_model=v07_model,
        device=device,
    )
    return triage_from_probabilities(v06f1_prob, v07_prob)


def validate_probabilities(
    v06f1_prob: float,
    v07_prob: float,
) -> tuple[float, float]:
    """Valida y recorta las probabilidades al intervalo unitario."""
    v06 = float(np.clip(v06f1_prob, 0.0, 1.0))
    v07 = float(np.clip(v07_prob, 0.0, 1.0))
    if not (np.isfinite(v06) and np.isfinite(v07)):
        raise ValueError("Probabilities must be finite numbers.")
    return v06, v07


__all__ = [
    "MiniResNetV06",
    "build_model",
    "load_checkpoint",
    "load_v06f1_model",
    "load_v07_model",
    "predict_two_model_probabilities",
    "triage_from_image",
    "triage_from_probabilities",
    "validate_probabilities",
    "preprocess_image",
    "FiveZoneTriageResult",
    "PreprocessingError",
    "InputImage",
]
