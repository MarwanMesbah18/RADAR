from dataclasses import dataclass
import cv2
import numpy as np
from PIL import Image
import config
from core.model_manager import ModelManager
from core.plate_utils import separate_chars


@dataclass
class CharDetection:
    bbox: tuple
    class_name: str
    confidence: float


@dataclass
class PlateOCRResult:
    text: str
    characters: list
    numbers: list
    yolo_detections: list
    annotated_image: np.ndarray = None
    confidence: float = 0.0


def _to_arabic(class_name):
    if class_name.isdigit():
        return class_name
    return config.FRANCO_TO_ARABIC.get(class_name, class_name)


def ocr_yolo(cropped_image, model_version=1, conf=None):
    """Run YOLO character detection on a cropped plate.

    Returns (List[CharDetection] sorted by x-position, annotated_image).
    model_version: 1 = V1, 2 = V2, 3 = V2 Weighted.
    conf: Optional confidence threshold override. Uses config.OCR_CONFIDENCE if None.
    """
    # Enhancement operations can produce non-contiguous arrays — YOLO needs contiguous
    if isinstance(cropped_image, np.ndarray):
        cropped_image = np.ascontiguousarray(cropped_image)

    mgr = ModelManager.get_instance()
    if model_version == 1:
        model = mgr.get_plate_ocr()
    elif model_version == 2:
        model = mgr.get_plate_ocr_v2()
    else:
        model = mgr.get_plate_ocr_v2w()
    threshold = conf if conf is not None else config.OCR_CONFIDENCE
    result = model.predict(
        source=cropped_image,
        conf=threshold,
        imgsz=640,
        verbose=False,
    )

    detections = []
    if not result or not result[0].boxes or len(result[0].boxes) == 0:
        img = np.array(cropped_image) if not isinstance(cropped_image, np.ndarray) else cropped_image
        return detections, img

    for box in result[0].boxes:
        xyxy = box.xyxy[0].tolist()
        class_id = int(box.cls[0])
        conf = box.conf[0].item()
        if class_id in result[0].names:
            detections.append(CharDetection(
                bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                class_name=_to_arabic(result[0].names[class_id]),
                confidence=conf,
            ))

    detections.sort(key=lambda d: d.bbox[0])

    annotated = result[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    return detections, annotated


def ocr_plate(cropped_image, model_version=1, conf=None):
    """Run YOLO OCR on a cropped plate. Returns PlateOCRResult."""
    yolo_detections, yolo_annotated = ocr_yolo(cropped_image, model_version=model_version, conf=conf)
    chars = separate_chars(yolo_detections)

    all_confs = [d.confidence for d in yolo_detections]
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

    return PlateOCRResult(
        text=chars.text,
        characters=chars.letters,
        numbers=chars.numbers,
        yolo_detections=yolo_detections,
        annotated_image=yolo_annotated,
        confidence=avg_conf,
    )
