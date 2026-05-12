from dataclasses import dataclass
from PIL import Image
import numpy as np
import config
from core.model_manager import ModelManager
from utils.preprocessing import crop_to_bbox


@dataclass
class PlateDetection:
    bbox: tuple          # (x1, y1, x2, y2) in original image coordinates
    confidence: float
    cropped_image: Image.Image


def detect_plates(image):
    """Detect license plates in an image.

    Args:
        image: PIL Image, numpy array (RGB), or file path string.

    Returns:
        List of PlateDetection objects, sorted by bbox width (widest first).
    """
    model = ModelManager.get_instance().get_plate_detector()
    results = model.predict(source=image, conf=config.PLATE_CONFIDENCE, verbose=False)

    # Get original image as PIL for cropping
    if isinstance(image, str):
        pil_image = Image.open(image)
    elif isinstance(image, Image.Image):
        pil_image = image
    else:
        pil_image = Image.fromarray(image)

    detections = []
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue

        for box in result.boxes:
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = xyxy.tolist()
            confidence = box.conf[0].item()
            width = x2 - x1

            cropped = crop_to_bbox(pil_image, (x1, y1, x2, y2),
                                   margin=config.PLATE_CROP_MARGIN)
            detections.append((width, PlateDetection(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                cropped_image=cropped,
            )))

    # Sort widest first (most likely the actual plate)
    detections.sort(key=lambda d: d[0], reverse=True)
    return [d[1] for d in detections]


def detect_plate_bbox(image):
    """Detect the single best (widest) plate bounding box in an image.

    Returns (x1, y1, x2, y2) or None.
    """
    plates = detect_plates(image)
    if plates:
        return plates[0].bbox
    return None
