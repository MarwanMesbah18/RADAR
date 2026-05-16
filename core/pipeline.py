from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import cv2
import time

import config
from core.car_tracker import track_cars
from core.plate_detector import detect_plates, PlateDetection
from core.plate_ocr import ocr_yolo, PlateOCRResult
from core.plate_utils import separate_chars
from core.seatbelt_detector import detect_seatbelt, get_seatbelt_summary, draw_seatbelt_detections
from core.enhancement import enhance_lapsrn, enhance_realesrgan
from core.preprocessing import pil_to_cv2, put_arabic_text


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
class SeatbeltMultiResult:
    original_detections: list = field(default_factory=list)
    lapsrn_detections: list = field(default_factory=list)
    esrgan_detections: list = field(default_factory=list)
    merged_detections: list = field(default_factory=list)
    summary: dict = None
    annotated_original: np.ndarray = None
    annotated_lapsrn: np.ndarray = None
    annotated_esrgan: np.ndarray = None


@dataclass
class VehicleAnalysis:
    car_index: int = 0
    car_class: str = ""
    plate_detection: PlateDetection = None
    plate_ocr: PlateOCRResult = None
    seatbelt_summary: dict = None
    seatbelt_multi: SeatbeltMultiResult = None
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


def run_multi_size_seatbelt(car_crop_bgr):
    """Run seatbelt detection at original, LapSRN, and Real-ESRGAN sizes."""
    dets_orig = detect_seatbelt(car_crop_bgr)
    annotated_orig = draw_seatbelt_detections(car_crop_bgr, dets_orig) if dets_orig else None

    dets_lap = []
    annotated_lap = None
    try:
        enhanced_lap = enhance_lapsrn(car_crop_bgr)
        if enhanced_lap is not None:
            dets_lap = detect_seatbelt(enhanced_lap)
            if dets_lap:
                annotated_lap = draw_seatbelt_detections(enhanced_lap, dets_lap)
    except Exception:
        pass

    dets_esrgan = []
    annotated_esrgan = None
    try:
        enhanced_esrgan = enhance_realesrgan(car_crop_bgr)
        if enhanced_esrgan is not None:
            dets_esrgan = detect_seatbelt(enhanced_esrgan)
            if dets_esrgan:
                annotated_esrgan = draw_seatbelt_detections(enhanced_esrgan, dets_esrgan)
    except Exception:
        pass

    # Merge: keep highest-confidence detection per class_id
    seen = {}
    for d in dets_orig + dets_lap + dets_esrgan:
        if d.class_id not in seen or d.confidence > seen[d.class_id].confidence:
            seen[d.class_id] = d
    merged = sorted(seen.values(), key=lambda d: d.confidence, reverse=True)
    summary = get_seatbelt_summary(merged)

    return SeatbeltMultiResult(
        original_detections=dets_orig,
        lapsrn_detections=dets_lap,
        esrgan_detections=dets_esrgan,
        merged_detections=merged,
        summary=summary,
        annotated_original=annotated_orig,
        annotated_lapsrn=annotated_lap,
        annotated_esrgan=annotated_esrgan,
    )


def generate_interior_summary(seatbelt_summary):
    """Generate human-readable summary of interior detection results."""
    if not seatbelt_summary:
        return "No interior analysis available."
    parts = []
    if seatbelt_summary.get("has_no_seatbelt"):
        parts.append("No seatbelt detected on driver.")
    elif seatbelt_summary.get("has_seatbelt"):
        parts.append("Seatbelt detected.")
    else:
        parts.append("Seatbelt status unclear.")
    if seatbelt_summary.get("has_mobile"):
        parts.append("Mobile phone in use.")
    else:
        parts.append("No mobile phone detected.")
    return " ".join(parts)


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

    # Run multi-size seatbelt detection (original + LapSRN + Real-ESRGAN)
    seatbelt_multi = run_multi_size_seatbelt(car_crop_bgr)
    seatbelt_summary = seatbelt_multi.summary
    if seatbelt_multi.annotated_original is not None:
        steps.seatbelt_annotated = seatbelt_multi.annotated_original

    # Detect plates in car crop
    plate_detections = detect_plates(search_pil)
    if not plate_detections:
        return VehicleAnalysis(
            car_index=car_index,
            car_class=car.class_name,
            seatbelt_summary=seatbelt_summary,
            seatbelt_multi=seatbelt_multi,
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
        seatbelt_multi=seatbelt_multi,
        steps=steps,
        yolo_detections=yolo_dets,
        yolo_numbers=chars.numbers,
        yolo_letters=chars.letters,
    )


def analyze_photo(image_input, multi_car=False):
    """Run analysis on a single image (kept for backward compatibility).

    Prefer using the step-by-step functions for progressive UI:
        detect_cars_step() -> analyze_interiors_step() -> analyze_plates_step()
    """
    start = time.time()
    pil_image, bgr_image = _load_image(image_input)
    if bgr_image is None:
        return PhotoAnalysisResult()

    h, w = bgr_image.shape[:2]
    result = PhotoAnalysisResult(original=bgr_image.copy())

    # Step 1: detect cars
    car_tracks, all_cars_image = detect_cars_step(bgr_image)
    result.all_cars_image = all_cars_image

    if not car_tracks:
        result.processing_time = time.time() - start
        return result

    if not multi_car:
        car_tracks = [max(car_tracks, key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]))]

    # Step 2: crop + seatbelt
    vehicles = analyze_interiors_step(bgr_image, car_tracks)

    # Step 3: plates + OCR
    plates_found = analyze_plates_step(bgr_image, pil_image, vehicles, w, h)

    result.vehicles = vehicles
    result.plates_found = plates_found
    result.cars_found = len(vehicles)
    result.processing_time = time.time() - start
    return result


# ── Step-by-step pipeline functions for progressive UI ──


def _load_image(image_input):
    """Load image from path, PIL, or numpy array. Returns (pil, bgr)."""
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
    return pil_image, bgr_image


def detect_cars_step(bgr_image, conf=None):
    """Step 1: Detect cars. Returns (car_tracks, annotated_image).

    Deduplicates overlapping detections using NMS (IoU > 0.4).
    conf: Optional confidence override. Uses config.CAR_CONFIDENCE if None.
    """
    car_tracks = track_cars(bgr_image, persist=False, conf=conf)

    # NMS: sort by area descending, greedily keep boxes that don't overlap
    if len(car_tracks) > 1:
        car_tracks.sort(
            key=lambda c: (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]),
            reverse=True,
        )
        kept = []
        for c in car_tracks:
            if not any(_bbox_iou(c.bbox, k.bbox) > 0.4 for k in kept):
                kept.append(c)
        car_tracks = kept

    annotated = bgr_image.copy()
    for i, car in enumerate(car_tracks):
        cx1, cy1, cx2, cy2 = [int(v) for v in car.bbox]
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
        cv2.putText(annotated, f"{car.class_name} #{i + 1}",
                    (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return car_tracks, annotated


def _bbox_iou(a, b):
    """IoU between two bboxes (x1,y1,x2,y2)."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1)


def analyze_interiors_step(bgr_image, car_tracks):
    """Step 2: Crop each car and run seatbelt detection at 3 sizes."""
    h, w = bgr_image.shape[:2]
    vehicles = []

    for i, car in enumerate(car_tracks):
        cx1, cy1, cx2, cy2 = [int(v) for v in car.bbox]
        crop_x1, crop_y1 = max(0, cx1), max(0, cy1)
        crop_x2, crop_y2 = min(w, cx2), min(h, cy2)
        car_crop = bgr_image[crop_y1:crop_y2, crop_x1:crop_x2]

        if car_crop.size == 0:
            continue

        car_marked = bgr_image.copy()
        cv2.rectangle(car_marked, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
        cv2.putText(car_marked, f"{car.class_name} #{i + 1}",
                    (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        steps = StepImages(car_detected=car_marked, car_crop=car_crop.copy())
        seatbelt_multi = run_multi_size_seatbelt(car_crop)
        if seatbelt_multi.annotated_original is not None:
            steps.seatbelt_annotated = seatbelt_multi.annotated_original

        vehicles.append(VehicleAnalysis(
            car_index=i,
            car_class=car.class_name,
            seatbelt_summary=seatbelt_multi.summary,
            seatbelt_multi=seatbelt_multi,
            steps=steps,
        ))

    return vehicles


def analyze_plates_step(bgr_image, pil_image, vehicles, frame_w, frame_h):
    """Step 3: Detect plates and run OCR on each vehicle."""
    plates_found = 0
    for v in vehicles:
        car_crop_rgb = cv2.cvtColor(v.steps.car_crop, cv2.COLOR_BGR2RGB)
        search_pil = Image.fromarray(car_crop_rgb)
        plate_detections = detect_plates(search_pil, conf=config.PLATE_CONFIDENCE_CACHE)

        if not plate_detections:
            continue

        plate_det = plate_detections[0]
        plate_crop_bgr = pil_to_cv2(plate_det.cropped_image)
        v.plate_detection = plate_det
        v.steps.plate_crop = plate_crop_bgr.copy()

        # OCR
        yolo_dets, _ = ocr_yolo(plate_det.cropped_image)
        chars = separate_chars(yolo_dets)

        yolo_display = plate_crop_bgr.copy()
        for d in yolo_dets:
            x1, y1, x2, y2 = [int(v2) for v2 in d.bbox]
            cv2.rectangle(yolo_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if d.class_name.isdigit():
                cv2.putText(yolo_display, d.class_name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                put_arabic_text(yolo_display, d.class_name, (x1, y1 - 20),
                                font_size=16, color=(0, 255, 0))
        v.steps.yolo_ocr = yolo_display

        all_confs = [d.confidence for d in yolo_dets]
        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
        v.plate_ocr = PlateOCRResult(
            text=chars.text, characters=chars.letters, numbers=chars.numbers,
            yolo_detections=yolo_dets, confidence=avg_conf,
        )
        v.yolo_detections = yolo_dets
        v.yolo_numbers = chars.numbers
        v.yolo_letters = chars.letters
        plates_found += 1

    return plates_found
