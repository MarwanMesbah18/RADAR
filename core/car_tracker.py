from dataclasses import dataclass
import numpy as np
import config
from core.model_manager import ModelManager


@dataclass
class CarTrack:
    track_id: int
    bbox: tuple          # (x1, y1, x2, y2)
    confidence: float
    class_name: str


def track_cars(frame, persist=True):
    """Detect and track cars in a frame using YOLO + ByteTrack.

    Args:
        frame: numpy array (BGR) of the video frame.
        persist: If True, maintain tracker state across calls.

    Returns:
        List[CarTrack] with track IDs for each detected vehicle.
    """
    model = ModelManager.get_instance().get_car_detector()

    results = model.track(
        source=frame,
        classes=config.CAR_CLASSES,
        conf=config.CAR_CONFIDENCE,
        tracker="bytetrack.yaml",
        persist=persist,
        verbose=False,
    )

    tracks = []
    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            track_id = int(box.id[0]) if box.id is not None else None
            if track_id is None:
                continue

            xyxy = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            class_id = int(box.cls[0])
            class_name = result.names.get(class_id, "vehicle")

            tracks.append(CarTrack(
                track_id=track_id,
                bbox=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                confidence=conf,
                class_name=class_name,
            ))

    return tracks
