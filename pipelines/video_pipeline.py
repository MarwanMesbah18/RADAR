import cv2
import tempfile
import time
from dataclasses import dataclass

import config
from core.car_tracker import track_cars
from core.plate_detector import detect_plates
from core.plate_ocr import ocr_yolo
from core.plate_aggregator import PlateAggregator, PlateReading
from utils.preprocessing import ensure_valid_bbox


@dataclass
class VideoProcessStats:
    total_frames: int = 0
    processed_frames: int = 0
    plates_detected: int = 0
    unique_plates: int = 0
    processing_fps: float = 0.0
    elapsed_time: float = 0.0


def _crop_car(frame, bbox):
    """Crop a car region from a frame."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return frame[y1:y2, x1:x2]


def _build_plate_text(detections):
    """Build plate text from sorted detections."""
    chars = []
    nums = []
    for det in detections:
        if det.class_name.isdigit():
            nums.append(det.class_name)
        else:
            chars.append(det.class_name)
    parts = []
    if nums:
        parts.append(" ".join(nums))
    if chars:
        parts.append(" ".join(chars))
    return " | ".join(parts)


def process_video(video_path, output_path=None, progress_callback=None, frame_skip=1):
    """Process a video file with car tracking → plate detection → OCR → deduplication.

    Args:
        video_path: Path to input video file.
        output_path: Path to save annotated output. If None, uses temp file.
        progress_callback: Callable(progress_fraction, stats_dict) called per frame.
        frame_skip: Process every Nth frame (1 = every frame).

    Returns:
        (output_path, VideoProcessStats, list[UniquePlate])
    """
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame += 1

        if frame_skip > 1 and current_frame % frame_skip != 0:
            out.write(frame)
            continue

        try:
            # Step 1: Detect and track cars
            car_tracks = track_cars(frame, persist=True)

            for car in car_tracks:
                # Draw car bounding box (blue)
                cx1, cy1, cx2, cy2 = [int(v) for v in car.bbox]
                cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (255, 150, 0), 2)
                cv2.putText(frame, f"Car #{car.track_id}", (cx1, cy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)

                # Step 2: Crop car region and detect plate inside
                car_crop = _crop_car(frame, car.bbox)
                if car_crop.size == 0:
                    continue

                # Convert car crop to PIL for plate detector
                from PIL import Image
                car_pil = Image.fromarray(cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB))
                plate_dets = detect_plates(car_pil)

                if not plate_dets:
                    continue

                plate = plate_dets[0]
                total_plates += 1

                # Convert plate bbox from car-crop coords to frame coords
                px1, py1, px2, py2 = [int(v) for v in plate.bbox]
                fx1 = cx1 + px1
                fy1 = cy1 + py1
                fx2 = cx1 + px2
                fy2 = cy1 + py2
                fx1, fy1, fx2, fy2 = ensure_valid_bbox(
                    (fx1, fy1, fx2, fy2), frame_width, frame_height
                )

                # Draw plate bounding box (green)
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                cv2.putText(frame, "Plate", (fx1, fy1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Step 3: OCR on the plate crop
                plate_crop = frame[fy1:fy2, fx1:fx2]
                if plate_crop.size == 0:
                    continue

                detections, _ = ocr_yolo(plate_crop)

                # Draw character boxes
                for det in detections:
                    dx1 = int(det.bbox[0]) + fx1
                    dy1 = int(det.bbox[1]) + fy1
                    dx2 = int(det.bbox[2]) + fx1
                    dy2 = int(det.bbox[3]) + fy1
                    cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 165, 255), 1)
                    cv2.putText(frame, det.class_name, (dx1, dy1 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

                plate_text = _build_plate_text(detections)

                if plate_text:
                    cv2.putText(frame, plate_text, (fx1, fy2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Calculate average OCR confidence
                avg_conf = (
                    sum(d.confidence for d in detections) / len(detections)
                    if detections else 0.0
                )

                # Step 4: Store reading for deduplication
                reading = PlateReading(
                    track_id=car.track_id,
                    frame_num=current_frame,
                    plate_text=plate_text,
                    confidence=avg_conf,
                    car_bbox=(cx1, cy1, cx2, cy2),
                    plate_bbox=(fx1, fy1, fx2, fy2),
                    frame_image=frame.copy(),
                )
                aggregator.add_reading(reading, car_class=car.class_name)

        except Exception as e:
            print(f"Error processing frame {current_frame}: {e}")

        out.write(frame)
        stats.processed_frames = current_frame
        stats.plates_detected = total_plates

        if progress_callback:
            progress_callback(current_frame / max(total_frames, 1), {
                "frame": current_frame,
                "total": total_frames,
                "plates": total_plates,
            })

    # Finalize: deduplicate plates by track
    unique_plates = aggregator.finalize()

    elapsed = time.time() - start_time
    stats.elapsed_time = elapsed
    stats.processing_fps = current_frame / elapsed if elapsed > 0 else 0
    stats.unique_plates = len(unique_plates)

    cap.release()
    out.release()

    return output_path, stats, unique_plates
