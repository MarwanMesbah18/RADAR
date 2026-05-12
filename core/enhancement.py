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


def _load_realesrgan():
    """Load Real-ESRGAN model (auto-downloads weights on first use)."""
    try:
        # Patch basicsr compatibility with newer torchvision
        import torchvision.transforms.functional as _F
        import sys
        if 'torchvision.transforms.functional_tensor' not in sys.modules:
            sys.modules['torchvision.transforms.functional_tensor'] = _F

        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        upsampler = RealESRGANer(
            scale=2,
            model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=False,
        )
        return upsampler
    except Exception as e:
        print(f"Real-ESRGAN load failed: {e}")
        return None


# Lazy-loaded singletons
_lapsrn_model = None
_realesrgan_model = None


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
        # Fallback: simple upscale + sharpen
        h, w = image.shape[:2]
        result = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
        return sharpen(result)

    result = _lapsrn_model.upsample(image)
    result = sharpen(result)
    return result


def enhance_realesrgan(image):
    """AI super-resolution using Real-ESRGAN (heavy, best quality)."""
    global _realesrgan_model
    if _realesrgan_model is None:
        _realesrgan_model = _load_realesrgan()
    if _realesrgan_model is None:
        return enhance_lapsrn(image)

    try:
        output, _ = _realesrgan_model.enhance(image, outscale=2)
        output = np.clip(output, 0, 255).astype(np.uint8)
        if output.mean() < 5:
            return enhance_lapsrn(image)
        return output
    except Exception:
        return enhance_lapsrn(image)


METHODS = {
    "lapsrn": ("LapSRN (AI Light)", enhance_lapsrn),
    "realesrgan": ("Real-ESRGAN (AI Heavy)", enhance_realesrgan),
}


def enhance(image, method="lapsrn"):
    """Apply enhancement to a plate crop."""
    fn = METHODS.get(method, METHODS["lapsrn"])[1]
    return fn(image)


def enhance_all(image):
    """Run all methods and return dict of results."""
    results = {"original": image}
    for key, (name, fn) in METHODS.items():
        try:
            results[key] = fn(image)
        except Exception:
            results[key] = image
    return results
