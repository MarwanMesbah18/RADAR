from dataclasses import dataclass
import numpy as np
import cv2
import config
from core.model_manager import ModelManager

SEATBELT_CLASS_NAMES = {
    0: "person-noseatbelt",
    1: "person-seatbelt",
    2: "seatbelt",
    3: "windshield",
    4: "mobile",
}


@dataclass
class SeatbeltDetection:
    bbox: tuple           # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str


def detect_seatbelt(image_bgr, conf=None):
    """Run seatbelt + mobile detection on an image crop.

    Args:
        image_bgr: numpy array (BGR) — typically a car or windshield crop.
        conf: Optional confidence threshold override. Uses config.SEATBELT_CONFIDENCE if None.

    Returns:
        List[SeatbeltDetection] sorted by confidence (highest first).
    """
    model = ModelManager.get_instance().get_seatbelt_detector()
    threshold = conf if conf is not None else config.SEATBELT_CONFIDENCE

    results = model.predict(
        source=image_bgr,
        imgsz=640,
        conf=threshold,
        verbose=False,
    )

    detections = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0])
            cls_name = SEATBELT_CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")
            detections.append(SeatbeltDetection(
                bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                confidence=conf,
                class_id=cls_id,
                class_name=cls_name,
            ))

    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def get_seatbelt_summary(detections):
    """Extract violation info from seatbelt detections.

    Returns dict with: has_seatbelt, has_no_seatbelt, has_mobile, annotated_image
    """
    has_seatbelt_obj = any(d.class_id == 2 for d in detections)
    has_seatbelt = any(d.class_id == 1 for d in detections) or has_seatbelt_obj
    has_no_seatbelt = any(d.class_id == 0 for d in detections) and not has_seatbelt_obj
    has_mobile = any(d.class_id == 4 for d in detections)

    return {
        "has_seatbelt": has_seatbelt,
        "has_no_seatbelt": has_no_seatbelt,
        "has_mobile": has_mobile,
        "detections": detections,
    }


def draw_seatbelt_detections(image_bgr, detections):
    """Draw colored bounding boxes for seatbelt/mobile detections."""
    colors = {
        0: (0, 0, 255),    # RED — no seatbelt
        1: (0, 200, 0),    # GREEN — seatbelt
        2: (255, 100, 0),  # BLUE — seatbelt object
        3: (0, 255, 255),  # YELLOW — windshield
        4: (255, 0, 255),  # MAGENTA — mobile
    }
    annotated = image_bgr.copy()
    h, w = annotated.shape[:2]

    font_scale = 0.7
    thickness = 2
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = colors.get(det.class_id, (128, 128, 128))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)

    return annotated
