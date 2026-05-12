from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image
import config
from core.model_manager import ModelManager


@dataclass
class CharDetection:
    bbox: tuple       # (x1, y1, x2, y2) relative to the cropped plate image
    class_name: str
    confidence: float


@dataclass
class PlateOCRResult:
    text: str                                    # Combined plate text
    characters: list                             # List of detected Arabic letters
    numbers: list                                # List of detected numbers
    yolo_detections: list                        # List[CharDetection]
    annotated_image: np.ndarray = None           # Annotated plate image
    confidence: float = 0.0


def _to_arabic(class_name):
    """Convert Franco class name to Arabic character, or return as-is for digits."""
    if class_name.isdigit():
        return class_name
    return config.FRANCO_TO_ARABIC.get(class_name, class_name)


def ocr_yolo(cropped_image):
    """Run YOLO-based character detection on a cropped plate image.

    Returns (List[CharDetection] sorted by x-position, annotated_image).
    """
    model = ModelManager.get_instance().get_plate_ocr()
    result = model.predict(source=cropped_image, conf=config.OCR_CONFIDENCE, verbose=False)

    detections = []
    if not result or not result[0].boxes or len(result[0].boxes) == 0:
        img = np.array(cropped_image) if not isinstance(cropped_image, np.ndarray) else cropped_image
        return detections, img

    boxes = result[0].boxes
    names = result[0].names

    for box in boxes:
        xyxy = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        conf = box.conf[0].item()

        if class_id in names:
            raw_name = names[class_id]
            detections.append(CharDetection(
                bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                class_name=_to_arabic(raw_name),
                confidence=conf,
            ))

    # Sort by x1 coordinate (left-to-right spatial order)
    detections.sort(key=lambda d: d.bbox[0])

    annotated = result[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return detections, annotated


def ocr_plate(cropped_image):
    """Run YOLO OCR pipeline on a cropped plate image.

    Args:
        cropped_image: PIL Image or numpy array of the cropped plate.

    Returns:
        PlateOCRResult with all detection data.
    """
    yolo_detections, yolo_annotated = ocr_yolo(cropped_image)

    numbers = []
    characters = []
    for det in yolo_detections:
        if det.class_name.isdigit():
            numbers.append(det.class_name)
        else:
            characters.append(det.class_name)

    # Build text from sorted detections (already sorted by x-position from ocr_yolo)
    text_parts = []
    if numbers:
        text_parts.append(" ".join(numbers))
    if characters:
        text_parts.append(" ".join(characters))
    combined_text = " | ".join(text_parts)

    all_confs = [d.confidence for d in yolo_detections]
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

    return PlateOCRResult(
        text=combined_text,
        characters=characters,
        numbers=numbers,
        yolo_detections=yolo_detections,
        annotated_image=yolo_annotated,
        confidence=avg_conf,
    )
