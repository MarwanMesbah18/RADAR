import cv2
import tempfile
import time
from dataclasses import dataclass, field
from PIL import Image

import config
from core.car_tracker import track_cars
from core.plate_detector import detect_plates
from core.plate_ocr import ocr_yolo
from core.plate_utils import separate_chars
from core.plate_aggregator import PlateAggregator, PlateReading
from core.enhancement import enhance
from core.seatbelt_detector import detect_seatbelt, get_seatbelt_summary
from utils.preprocessing import ensure_valid_bbox, put_arabic_text


@dataclass
class TrackedCar:
    track_id: int
    car_class: str
    best_crop: object = None  # numpy array
    best_frame_num: int = 0
    first_seen: int = 0


@dataclass
class VideoProcessStats:
    total_frames: int = 0
    processed_frames: int = 0
    plates_detected: int = 0
    unique_plates: int = 0
    total_cars: int = 0
    processing_fps: float = 0.0
    elapsed_time: float = 0.0


def _crop_car(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]


def process_video(video_path, output_path=None, progress_callback=None,
                  result_callback=None, frame_skip=1, enhance_method="combined"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        output_path = tmp.name
        tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    aggregator = PlateAggregator()
    stats = VideoProcessStats(total_frames=total_frames)
    start_time = time.time()
    current_frame = 0
    total_plates = 0
    good_tracks: set[int] = set()

    # Track ALL cars (with and without plates)
    all_cars: dict[int, TrackedCar] = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1

        if frame_skip > 1 and current_frame % frame_skip != 0:
            out.write(frame)
            continue

        try:
            car_tracks = track_cars(frame, persist=True)

            for car in car_tracks:
                cx1, cy1, cx2, cy2 = [int(v) for v in car.bbox]
                car_crop = _crop_car(frame, car.bbox)

                # Always draw car box
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
                cv2.putText(frame, f"Car #{car.track_id}", (cx1, cy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Track this car (save best crop)
                if car.track_id not in all_cars:
                    all_cars[car.track_id] = TrackedCar(
                        track_id=car.track_id,
                        car_class=car.class_name,
                        first_seen=current_frame,
                    )
                if car_crop.size > 0:
                    car_obj = all_cars[car.track_id]
                    prev_area = car_obj.best_crop.shape[0] * car_obj.best_crop.shape[1] if car_obj.best_crop is not None else 0
                    curr_area = car_crop.shape[0] * car_crop.shape[1]
                    if curr_area > prev_area:
                        car_obj.best_crop = car_crop.copy()
                        car_obj.best_frame_num = current_frame

                # Skip OCR if already has good reading
                if car.track_id in good_tracks:
                    continue

                if car_crop.size == 0:
                    continue

                car_pil = Image.fromarray(cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB))
                plate_dets = detect_plates(car_pil)
                if not plate_dets:
                    continue

                plate = plate_dets[0]
                total_plates += 1

                px1, py1, px2, py2 = [int(v) for v in plate.bbox]
                fx1, fy1 = cx1 + px1, cy1 + py1
                fx2, fy2 = cx1 + px2, cy1 + py2
                fx1, fy1, fx2, fy2 = ensure_valid_bbox(
                    (fx1, fy1, fx2, fy2), frame_width, frame_height
                )

                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                cv2.putText(frame, "Plate", (fx1, fy1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                plate_crop = frame[fy1:fy2, fx1:fx2]
                if plate_crop.size == 0:
                    continue

                # Apply enhancement before OCR
                enhanced_crop = enhance(plate_crop, method=enhance_method)

                detections, ocr_annotated = ocr_yolo(enhanced_crop)
                chars = separate_chars(detections)

                avg_conf = (
                    sum(d.confidence for d in detections) / len(detections)
                    if detections else 0.0
                )

                # Draw annotations
                for det in detections:
                    dx1 = int(det.bbox[0]) + fx1
                    dy1 = int(det.bbox[1]) + fy1
                    dx2 = int(det.bbox[2]) + fx1
                    dy2 = int(det.bbox[3]) + fy1
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 1)
                    if det.class_name.isdigit():
                        cv2.putText(frame, det.class_name, (dx1, dy1 - 3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    else:
                        put_arabic_text(frame, det.class_name, (dx1, dy1 - 18),
                                        font_size=12, color=(0, 255, 0))

                if chars.text:
                    put_arabic_text(frame, chars.text, (fx1, fy2 + 5),
                                    font_size=18, color=(0, 255, 0))

                if avg_conf >= config.VIDEO_OCR_MIN_CONFIDENCE:
                    reading = PlateReading(
                        track_id=car.track_id,
                        frame_num=current_frame,
                        plate_text=chars.text,
                        confidence=avg_conf,
                        car_bbox=(cx1, cy1, cx2, cy2),
                        plate_bbox=(fx1, fy1, fx2, fy2),
                        car_crop_image=car_crop.copy(),
                        plate_crop_image=plate_crop.copy(),
                        ocr_annotated_image=ocr_annotated.copy() if ocr_annotated is not None else None,
                        detections=detections,
                    )
                    aggregator.add_reading(reading, car_class=car.class_name)
                    good_tracks.add(car.track_id)

                    if result_callback:
                        result_callback(reading, car.class_name)

        except Exception as e:
            print(f"Error processing frame {current_frame}: {e}")

        out.write(frame)
        stats.processed_frames = current_frame
        stats.plates_detected = total_plates
        stats.total_cars = len(all_cars)

        if progress_callback:
            progress_callback(current_frame / max(total_frames, 1), {
                "frame": current_frame,
                "total": total_frames,
                "plates": total_plates,
                "unique": len(good_tracks),
                "cars": len(all_cars),
            })

    unique_plates = aggregator.finalize()

    # Separate cars with and without plates
    plate_track_ids = {p.car_track_id for p in unique_plates}
    cars_without_plates = []
    for tid, car in all_cars.items():
        if tid not in plate_track_ids:
            cars_without_plates.append(car)
    cars_without_plates.sort(key=lambda c: c.first_seen)

    # Attach best frame images for plates
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    best_frames = {p.best_frame_num: p for p in unique_plates}
    for frame_idx in sorted(best_frames.keys()):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - 1)
        ret, frame = cap.read()
        if ret:
            best_frames[frame_idx].best_frame_image = frame.copy()

    elapsed = time.time() - start_time
    stats.elapsed_time = elapsed
    stats.processing_fps = current_frame / elapsed if elapsed > 0 else 0
    stats.unique_plates = len(unique_plates)

    cap.release()
    out.release()

    return output_path, stats, unique_plates, cars_without_plates


# ── Interactive video pipeline functions ──


@dataclass
class ScannedCar:
    track_id: int
    car_class: str
    best_crop: object = None  # numpy array
    best_frame_num: int = 0
    first_seen_frame: int = 0
    best_bbox: tuple = None   # (x1, y1, x2, y2) in frame coords
    seatbelt_summary: dict = None
    all_bboxes: dict = field(default_factory=dict)     # frame_num -> bbox
    frames_visible: set = field(default_factory=set)    # frames where car appeared
    plate_candidates: list = field(default_factory=list)  # List[PlateCandidate]
    has_plate: bool = False


@dataclass
class PlateCandidate:
    frame_num: int
    plate_crop: object = None  # numpy array (BGR)
    plate_bbox: tuple = None   # (x1, y1, x2, y2) in frame coords
    car_crop: object = None    # numpy array (BGR)
    crop_area: float = 0.0


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


def scan_video_cars(video_path, progress_callback=None):
    """Scan a video to find all unique cars via ByteTrack tracking.

    Returns list of ScannedCar sorted by first_seen_frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_cars: dict[int, ScannedCar] = {}
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        try:
            car_tracks = track_cars(frame, persist=True)

            for car in car_tracks:
                car_crop = _crop_car(frame, car.bbox)
                if car_crop.size == 0:
                    continue

                if car.track_id not in all_cars:
                    all_cars[car.track_id] = ScannedCar(
                        track_id=car.track_id,
                        car_class=car.class_name,
                        first_seen_frame=current_frame,
                    )

                car_obj = all_cars[car.track_id]
                curr_area = car_crop.shape[0] * car_crop.shape[1]
                prev_area = car_obj.best_crop.shape[0] * car_obj.best_crop.shape[1] if car_obj.best_crop is not None else 0
                if curr_area > prev_area:
                    car_obj.best_crop = car_crop.copy()
                    car_obj.best_frame_num = current_frame
                    car_obj.best_bbox = tuple(int(v) for v in car.bbox)
        except Exception as e:
            print(f"Error scanning frame {current_frame}: {e}")

        if progress_callback:
            progress_callback(current_frame / max(total_frames, 1), {
                "frame": current_frame,
                "total": total_frames,
                "cars": len(all_cars),
            })

    # Deduplicate: only merge tracks that overlap heavily (same car, lost ID)
    car_list = list(all_cars.values())
    merged = []
    used = set()
    for i, c1 in enumerate(car_list):
        if i in used:
            continue
        group = [c1]
        for j in range(i + 1, len(car_list)):
            if j in used:
                continue
            c2 = car_list[j]
            # Only merge if same class AND frames overlap (both visible at same time)
            # AND bboxes nearly identical (same position = same car with split ID)
            if (c1.car_class == c2.car_class
                    and c1.best_bbox and c2.best_bbox
                    and _bbox_iou(c1.best_bbox, c2.best_bbox) > 0.5
                    and abs(c1.best_frame_num - c2.best_frame_num) < 30):
                group.append(c2)
                used.add(j)
        # Pick the one with largest crop
        best_in_group = max(group, key=lambda c: c.best_crop.shape[0] * c.best_crop.shape[1] if c.best_crop is not None else 0)
        merged.append(best_in_group)
        used.add(i)

    cap.release()
    cars = sorted(merged, key=lambda c: c.first_seen_frame)

    # Run seatbelt detection on each car's best crop
    for car in cars:
        if car.best_crop is not None and car.best_crop.size > 0:
            seatbelt_dets = detect_seatbelt(car.best_crop)
            car.seatbelt_summary = get_seatbelt_summary(seatbelt_dets)

    return cars


def find_plate_crops(video_path, scanned_car, top_n=5):
    """Find plate crop candidates for a specific scanned car.

    Matches by spatial proximity to scanned_car.best_bbox (not track ID),
    since ByteTrack assigns fresh IDs on each pass.

    scanned_car: ScannedCar object with best_bbox set.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    candidates: list[PlateCandidate] = []
    ref_bbox = scanned_car.best_bbox

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        try:
            car_tracks = track_cars(frame, persist=True)

            for car in car_tracks:
                # Match by position, not track ID
                if ref_bbox is not None:
                    car_bbox = tuple(int(v) for v in car.bbox)
                    if _bbox_iou(ref_bbox, car_bbox) < 0.15:
                        continue

                if car.class_name != scanned_car.car_class:
                    continue

                car_crop = _crop_car(frame, car.bbox)
                if car_crop.size == 0:
                    continue

                cx1, cy1 = int(car.bbox[0]), int(car.bbox[1])
                car_pil = Image.fromarray(cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB))
                plate_dets = detect_plates(car_pil)
                if not plate_dets:
                    continue

                plate = plate_dets[0]
                px1, py1, px2, py2 = [int(v) for v in plate.bbox]

                # Plate crop in frame coordinates
                fx1, fy1 = cx1 + px1, cy1 + py1
                fx2, fy2 = cx1 + px2, cy1 + py2
                fx1, fy1, fx2, fy2 = ensure_valid_bbox(
                    (fx1, fy1, fx2, fy2), frame_width, frame_height
                )

                plate_crop = frame[fy1:fy2, fx1:fx2]
                if plate_crop.size == 0:
                    continue

                crop_area = (fx2 - fx1) * (fy2 - fy1)

                # Skip if very similar frame already captured (deduplicate by area proximity)
                is_dup = any(abs(c.crop_area - crop_area) / max(crop_area, 1) < 0.1 for c in candidates)
                if is_dup:
                    continue

                candidates.append(PlateCandidate(
                    frame_num=current_frame,
                    plate_crop=plate_crop.copy(),
                    plate_bbox=(fx1, fy1, fx2, fy2),
                    car_crop=car_crop.copy(),
                    crop_area=crop_area,
                ))
        except Exception as e:
            print(f"Error finding plates at frame {current_frame}: {e}")

    cap.release()

    # Sort by crop area descending, take top_n
    candidates.sort(key=lambda c: c.crop_area, reverse=True)
    return candidates[:top_n]


# ── Single-pass scan: cars + plates in one pass ──


def scan_video_full(video_path, progress_callback=None):
    """Single-pass scan: find all cars AND their plate candidates.

    Replaces the old two-pass approach (scan_video_cars + find_plate_crops).
    During tracking, plate detection runs on each car crop so plates are
    naturally associated with the correct car — no spatial re-matching needed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_cars: dict[int, ScannedCar] = {}
    current_frame = 0
    total_plates = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        try:
            car_tracks = track_cars(frame, persist=True)

            for car in car_tracks:
                car_crop = _crop_car(frame, car.bbox)
                if car_crop.size == 0:
                    continue

                car_bbox = tuple(int(v) for v in car.bbox)

                if car.track_id not in all_cars:
                    all_cars[car.track_id] = ScannedCar(
                        track_id=car.track_id,
                        car_class=car.class_name,
                        first_seen_frame=current_frame,
                        all_bboxes={},
                        frames_visible=set(),
                        plate_candidates=[],
                    )

                car_obj = all_cars[car.track_id]
                car_obj.all_bboxes[current_frame] = car_bbox
                car_obj.frames_visible.add(current_frame)

                # Update best crop (largest)
                curr_area = car_crop.shape[0] * car_crop.shape[1]
                prev_area = car_obj.best_crop.shape[0] * car_obj.best_crop.shape[1] if car_obj.best_crop is not None else 0
                if curr_area > prev_area:
                    car_obj.best_crop = car_crop.copy()
                    car_obj.best_frame_num = current_frame
                    car_obj.best_bbox = car_bbox

                # Skip plate detection if already have enough candidates
                if len(car_obj.plate_candidates) >= config.MAX_PLATE_CANDIDATES_PER_CAR:
                    continue

                # Run plate detection on this car crop
                car_pil = Image.fromarray(cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB))
                plate_dets = detect_plates(car_pil)
                if not plate_dets:
                    continue

                plate = plate_dets[0]
                px1, py1, px2, py2 = [int(v) for v in plate.bbox]
                cx1, cy1 = int(car.bbox[0]), int(car.bbox[1])
                fx1, fy1 = cx1 + px1, cy1 + py1
                fx2, fy2 = cx1 + px2, cy1 + py2
                fx1, fy1, fx2, fy2 = ensure_valid_bbox(
                    (fx1, fy1, fx2, fy2), frame_width, frame_height
                )

                plate_crop = frame[fy1:fy2, fx1:fx2]
                if plate_crop.size == 0:
                    continue

                crop_area = (fx2 - fx1) * (fy2 - fy1)

                # Deduplicate by area similarity
                is_dup = any(
                    abs(c.crop_area - crop_area) / max(crop_area, 1) < config.PLATE_DEDUP_AREA_TOLERANCE
                    for c in car_obj.plate_candidates
                )
                if is_dup:
                    continue

                car_obj.plate_candidates.append(PlateCandidate(
                    frame_num=current_frame,
                    plate_crop=plate_crop.copy(),
                    plate_bbox=(fx1, fy1, fx2, fy2),
                    car_crop=car_crop.copy(),
                    crop_area=crop_area,
                ))
                car_obj.has_plate = True
                total_plates += 1

        except Exception as e:
            print(f"Error scanning frame {current_frame}: {e}")

        if progress_callback:
            progress_callback(current_frame / max(total_frames, 1), {
                "frame": current_frame,
                "total": total_frames,
                "cars": len(all_cars),
                "plates": total_plates,
            })

    cap.release()

    # Deduplicate car tracks (merge split track IDs belonging to same car)
    car_list = list(all_cars.values())
    merged = _deduplicate_cars(car_list)

    # Sort plate candidates by area (largest first), limit per car
    for car in merged:
        car.plate_candidates.sort(key=lambda c: c.crop_area, reverse=True)
        car.plate_candidates = car.plate_candidates[:config.MAX_PLATE_CANDIDATES_PER_CAR]

    # Run seatbelt detection on each car's best crop
    for car in merged:
        if car.best_crop is not None and car.best_crop.size > 0:
            seatbelt_dets = detect_seatbelt(car.best_crop)
            car.seatbelt_summary = get_seatbelt_summary(seatbelt_dets)

    return sorted(merged, key=lambda c: c.first_seen_frame)


def _deduplicate_cars(car_list):
    """Merge car tracks that belong to the same physical car.

    Key improvement: cars visible in the same frame are NEVER merged
    (they are genuinely different cars). Only merge tracks of same class
    with high IoU that were never seen simultaneously.
    """
    merged = []
    used = set()

    for i, c1 in enumerate(car_list):
        if i in used:
            continue
        group = [c1]
        for j in range(i + 1, len(car_list)):
            if j in used:
                continue
            c2 = car_list[j]
            # Must be same class
            if c1.car_class != c2.car_class:
                continue
            # Never merge cars visible in the same frame
            if c1.frames_visible & c2.frames_visible:
                continue
            # Must have high bbox IoU
            if not (c1.best_bbox and c2.best_bbox):
                continue
            if _bbox_iou(c1.best_bbox, c2.best_bbox) < config.CAR_MERGE_IOU_THRESHOLD:
                continue
            group.append(c2)
            used.add(j)

        # Pick best crop from group
        best = max(group, key=lambda c: c.best_crop.shape[0] * c.best_crop.shape[1] if c.best_crop is not None else 0)

        # Merge plate candidates from all tracks in group
        all_plates = []
        for c in group:
            all_plates.extend(c.plate_candidates)

        # Deduplicate merged plates by area
        seen_areas = set()
        unique_plates = []
        for p in sorted(all_plates, key=lambda p: p.crop_area, reverse=True):
            area_key = round(p.crop_area, -1)
            if area_key not in seen_areas:
                unique_plates.append(p)
                seen_areas.add(area_key)

        best.plate_candidates = unique_plates[:config.MAX_PLATE_CANDIDATES_PER_CAR]
        best.has_plate = len(best.plate_candidates) > 0

        # Merge trajectory data
        for c in group:
            if c is not best:
                best.all_bboxes.update(c.all_bboxes)
                best.frames_visible.update(c.frames_visible)

        merged.append(best)
        used.add(i)

    return merged
