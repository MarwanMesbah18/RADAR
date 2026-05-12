from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import cv2
import time

from core.plate_detector import detect_plates, PlateDetection
from core.plate_ocr import ocr_yolo, PlateOCRResult, CharDetection
from utils.preprocessing import pil_to_cv2


@dataclass
class StepImages:
    """Intermediate images captured at each analysis step (all BGR)."""
    original: np.ndarray = None
    plate_detected: np.ndarray = None
    plate_crop: np.ndarray = None
    yolo_ocr: np.ndarray = None


@dataclass
class VehicleAnalysis:
    plate_detection: PlateDetection = None
    plate_ocr: PlateOCRResult = None
    yolo_detections: list = field(default_factory=list)
    yolo_numbers: list = field(default_factory=list)
    yolo_letters: list = field(default_factory=list)


@dataclass
class PhotoAnalysisResult:
    steps: StepImages = None
    vehicles: list = field(default_factory=list)
    processing_time: float = 0.0
    plates_found: int = 0


def analyze_photo(image_input):
    """Run full analysis pipeline on a single image."""
    start = time.time()
    steps = StepImages()

    # Step 1: Load image
    if isinstance(image_input, str):
        pil_image = Image.open(image_input).convert("RGB")
        bgr_image = cv2.imread(image_input)
    elif isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")
        bgr_image = pil_to_cv2(pil_image)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            bgr_image = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        elif image_input.shape[2] == 4:
            bgr_image = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
        else:
            bgr_image = image_input
        pil_image = Image.fromarray(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB))
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    steps.original = bgr_image.copy()

    # Step 2: Detect plates
    plate_detections = detect_plates(pil_image)

    if not plate_detections:
        elapsed = time.time() - start
        return PhotoAnalysisResult(steps=steps, processing_time=elapsed, plates_found=0)

    plate_det = plate_detections[0]
    analysis = VehicleAnalysis(plate_detection=plate_det)

    # Step 2 image: original with plate bbox drawn
    plate_marked = bgr_image.copy()
    bx1, by1, bx2, by2 = [int(v) for v in plate_det.bbox]
    cv2.rectangle(plate_marked, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    cv2.putText(plate_marked, f"Plate {plate_det.confidence:.0%}", (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    steps.plate_detected = plate_marked

    # Step 3: Cropped plate
    plate_crop_bgr = pil_to_cv2(plate_det.cropped_image)
    steps.plate_crop = plate_crop_bgr.copy()

    # Step 4: YOLO OCR (detections already sorted by x-position, class names already Arabic)
    yolo_dets, yolo_annotated = ocr_yolo(plate_det.cropped_image)
    analysis.yolo_detections = yolo_dets

    yolo_numbers = []
    yolo_letters = []
    for d in yolo_dets:
        if d.class_name.isdigit():
            yolo_numbers.append(d.class_name)
        else:
            yolo_letters.append(d.class_name)
    analysis.yolo_numbers = yolo_numbers
    analysis.yolo_letters = yolo_letters

    yolo_display = plate_crop_bgr.copy()
    for d in yolo_dets:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        cv2.rectangle(yolo_display, (x1, y1), (x2, y2), (255, 150, 0), 2)
        cv2.putText(yolo_display, d.class_name, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
    steps.yolo_ocr = yolo_display

    # Build result
    text_parts = []
    if yolo_numbers:
        text_parts.append(" ".join(yolo_numbers))
    if yolo_letters:
        text_parts.append(" ".join(yolo_letters))
    combined_text = " | ".join(text_parts)

    all_confs = [d.confidence for d in yolo_dets]
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

    analysis.plate_ocr = PlateOCRResult(
        text=combined_text,
        characters=yolo_letters,
        numbers=yolo_numbers,
        yolo_detections=yolo_dets,
        confidence=avg_conf,
    )

    elapsed = time.time() - start

    return PhotoAnalysisResult(
        steps=steps,
        vehicles=[analysis],
        processing_time=elapsed,
        plates_found=len(plate_detections),
    )
