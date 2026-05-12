import cv2
import numpy as np
from PIL import Image
import config


def apply_morphological_operations(image):
    """Apply morphological preprocessing to enhance text for OCR."""
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image

    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    _, thresh = cv2.threshold(gray, config.BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones(config.MORPH_KERNEL_SIZE, np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE, kernel)

    return Image.fromarray(closed)


def crop_to_bbox(image, bbox, margin=0):
    """Crop an image to the given bounding box with optional margin."""
    x1, y1, x2, y2 = bbox
    if isinstance(image, Image.Image):
        w, h = image.size
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1))
        x2 = min(w, int(x2) + margin)
        y2 = min(h, int(y2))
        return image.crop((x1, y1, x2, y2))
    else:
        h, w = image.shape[:2]
        x1 = max(0, int(x1) - margin)
        y1 = max(0, int(y1))
        x2 = min(w, int(x2) + margin)
        y2 = min(h, int(y2))
        return image[y1:y2, x1:x2]


def ensure_valid_bbox(bbox, max_width, max_height):
    """Clamp bounding box coordinates to image bounds."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(max_width, x2)
    y2 = min(max_height, y2)
    return x1, y1, x2, y2


def pil_to_cv2(image):
    """Convert PIL Image to OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image):
    """Convert OpenCV BGR numpy array to PIL Image."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
