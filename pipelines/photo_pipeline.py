from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import cv2
import time

from core.car_tracker import track_cars
from core.plate_detector import detect_plates, PlateDetection
from core.plate_ocr import ocr_yolo, PlateOCRResult
from core.plate_utils import separate_chars
from core.seatbelt_detector import detect_seatbelt, get_seatbelt_summary, draw_seatbelt_detections
from utils.preprocessing import pil_to_cv2, put_arabic_text


@dataclass
class StepImages:
    """Intermediate images for a single vehicle's analysis (all BGR)."""
    original: np.ndarray = None
    car_detected: np.ndarray = None
    car_crop: np.ndarray = None
    seatbelt_annotated: np.ndarray = None
    plate_detected: np.ndarray = None
    plate_crop: np.ndarray = None
    yolo_ocr: np.ndarray = None


@dataclass
class VehicleAnalysis:
    car_index: int = 0
    car_class: str = ""
    plate_detection: PlateDetection = None
    plate_ocr: PlateOCRResult = None
    seatbelt_summary: dict = None
    steps: StepImages = None
    yolo_detections: list = field(default_factory=list)
    yolo_numbers: list = field(default_factory=list)
    yolo_letters: list = field(default_factory=list)


@dataclass
class PhotoAnalysisResult:
    original: np.ndarray = None
    all_cars_image: np.ndarray = None
    vehicles: list = field(default_factory=list)
    processing_time: float = 0.0
    plates_found: int = 0
    cars_found: int = 0


def _analyze_single_car(bgr_image, pil_image, car, car_index, frame_width, frame_height):
    """Run the full pipeline on a single detected car. Returns VehicleAnalysis or None."""
    cx1, cy1, cx2, cy2 = [int(v) for v in car.bbox]

    # Mark car on original
    car_marked = bgr_image.copy()
    cv2.rectangle(car_marked, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
    cv2.putText(car_marked, f"{car.class_name} #{car_index + 1}",
                (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Crop car region
    h, w = bgr_image.shape[:2]
    crop_x1, crop_y1 = max(0, cx1), max(0, cy1)
    crop_x2, crop_y2 = min(w, cx2), min(h, cy2)
    car_crop_bgr = bgr_image[crop_y1:crop_y2, crop_x1:crop_x2]

    if car_crop_bgr.size == 0:
        return None

    search_pil = Image.fromarray(cv2.cvtColor(car_crop_bgr, cv2.COLOR_BGR2RGB))

    steps = StepImages(
        car_detected=car_marked,
        car_crop=car_crop_bgr.copy(),
    )

    # Run seatbelt + mobile detection on car crop
    seatbelt_dets = detect_seatbelt(car_crop_bgr)
    seatbelt_summary = get_seatbelt_summary(seatbelt_dets)
    if seatbelt_dets:
        steps.seatbelt_annotated = draw_seatbelt_detections(car_crop_bgr, seatbelt_dets)

    # Detect plates in car crop
    plate_detections = detect_plates(search_pil)
    if not plate_detections:
        return VehicleAnalysis(
            car_index=car_index,
            car_class=car.class_name,
            seatbelt_summary=seatbelt_summary,
            steps=steps,
        )

    plate_det = plate_detections[0]

    # Draw plate bbox (translate from car-crop coords to frame coords)
    plate_display = car_marked.copy()
    px1, py1, px2, py2 = [int(v) for v in plate_det.bbox]
    px1 += cx1; py1 += cy1; px2 += cx1; py2 += cy1
    cv2.rectangle(plate_display, (px1, py1), (px2, py2), (0, 255, 0), 2)
    cv2.putText(plate_display, f"Plate {plate_det.confidence:.0%}",
                (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    steps.plate_detected = plate_display

    # Crop plate
    plate_crop_bgr = pil_to_cv2(plate_det.cropped_image)
    steps.plate_crop = plate_crop_bgr.copy()

    # OCR
    yolo_dets, yolo_annotated = ocr_yolo(plate_det.cropped_image)
    chars = separate_chars(yolo_dets)

    # Draw character boxes (green)
    yolo_display = plate_crop_bgr.copy()
    for d in yolo_dets:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        cv2.rectangle(yolo_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if d.class_name.isdigit():
            cv2.putText(yolo_display, d.class_name, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            put_arabic_text(yolo_display, d.class_name, (x1, y1 - 20),
                            font_size=16, color=(0, 255, 0))
    steps.yolo_ocr = yolo_display

    all_confs = [d.confidence for d in yolo_dets]
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0

    plate_ocr = PlateOCRResult(
        text=chars.text,
        characters=chars.letters,
        numbers=chars.numbers,
        yolo_detections=yolo_dets,
        confidence=avg_conf,
    )

    return VehicleAnalysis(
        car_index=car_index,
        car_class=car.class_name,
        plate_detection=plate_det,
        plate_ocr=plate_ocr,
        seatbelt_summary=seatbelt_summary,
        steps=steps,
        yolo_detections=yolo_dets,
        yolo_numbers=chars.numbers,
        yolo_letters=chars.letters,
    )


def analyze_photo(image_input, multi_car=False):
    """Run analysis on a single image.

    Args:
        image_input: file path, PIL Image, or numpy array.
        multi_car: If True, analyze ALL detected cars. If False, only the biggest car.

    Returns:
        PhotoAnalysisResult
    """
    start = time.time()

    # Load image
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

    h, w = bgr_image.shape[:2]

    # Detect cars
    car_tracks = track_cars(bgr_image, persist=False)

    if not car_tracks:
        # No cars — try plate detection on full image
        elapsed = time.time() - start
        return PhotoAnalysisResult(
            original=bgr_image.copy(),
            processing_time=elapsed,
            cars_found=0,
        )

    # Draw all cars on one image
    all_cars = bgr_image.copy()
    for i, car in enumerate(car_tracks):
        cx1, cy1, cx2, cy2 = [int(v) for v in car.bbox]
        cv2.rectangle(all_cars, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
        cv2.putText(all_cars, f"{car.class_name} #{i + 1}",
                    (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Pick which cars to analyze
    if multi_car:
        cars_to_analyze = car_tracks
    else:
        # Single mode: pick the largest car
        best = max(car_tracks, key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]))
        cars_to_analyze = [best]

    # Analyze each car
    vehicles = []
    plates_found = 0
    for i, car in enumerate(cars_to_analyze):
        result = _analyze_single_car(bgr_image, pil_image, car, i, w, h)
        if result is not None:
            vehicles.append(result)
            if result.plate_detection is not None:
                plates_found += 1

    elapsed = time.time() - start

    return PhotoAnalysisResult(
        original=bgr_image.copy(),
        all_cars_image=all_cars,
        vehicles=vehicles,
        processing_time=elapsed,
        plates_found=plates_found,
        cars_found=len(cars_to_analyze),
    )
