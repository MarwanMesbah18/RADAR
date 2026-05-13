import os
import cv2
import numpy as np
import config


def _load_lapsrn():
    """Load LapSRN x2 model via OpenCV DNN."""
    model_path = os.path.join(config.MODELS_DIR, "LapSRN_x2.pb")
    if not os.path.exists(model_path):
        return None
    try:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("lapsrn", 2)
        return sr
    except Exception:
        return None


# Lazy-loaded singleton
_lapsrn_model = None


def sharpen(image):
    """Unsharp mask sharpening."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def enhance_lapsrn(image):
    """AI super-resolution using LapSRN (lightweight, ~1.3MB model)."""
    global _lapsrn_model
    if _lapsrn_model is None:
        _lapsrn_model = _load_lapsrn()
    if _lapsrn_model is None:
        h, w = image.shape[:2]
        result = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        return sharpen(result)

    try:
        result = _lapsrn_model.upsample(image)
        result = sharpen(result)
        return result
    except cv2.error:
        h, w = image.shape[:2]
        result = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        return sharpen(result)


METHODS = {
    "lapsrn": ("LapSRN (AI Light)", enhance_lapsrn),
}


def enhance(image, method="lapsrn"):
    """Apply enhancement to a plate crop."""
    fn = METHODS.get(method, METHODS["lapsrn"])[1]
    return fn(image)
