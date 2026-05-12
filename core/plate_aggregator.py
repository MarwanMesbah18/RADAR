from dataclasses import dataclass, field
import numpy as np


@dataclass
class PlateReading:
    track_id: int
    frame_num: int
    plate_text: str
    confidence: float
    car_bbox: tuple            # (x1, y1, x2, y2) in frame coords
    plate_bbox: tuple          # (x1, y1, x2, y2) in frame coords
    frame_image: np.ndarray = None  # Copy of the annotated frame


@dataclass
class UniquePlate:
    plate_text: str
    best_confidence: float
    best_frame_num: int
    best_frame_image: np.ndarray = None
    car_track_id: int = -1
    car_class: str = ""
    total_detections: int = 0


class PlateAggregator:
    """Collects plate readings across video frames and deduplicates by car track."""

    def __init__(self):
        self._readings: dict[int, list[PlateReading]] = {}
        self._car_classes: dict[int, str] = {}

    def add_reading(self, reading: PlateReading, car_class: str = ""):
        tid = reading.track_id
        if tid not in self._readings:
            self._readings[tid] = []
            self._car_classes[tid] = car_class
        self._readings[tid].append(reading)

    def finalize(self) -> list[UniquePlate]:
        """Group readings by track_id and pick the best reading per car."""
        unique_plates = []

        for track_id, readings in self._readings.items():
            best = max(readings, key=lambda r: r.confidence)
            unique_plates.append(UniquePlate(
                plate_text=best.plate_text,
                best_confidence=best.confidence,
                best_frame_num=best.frame_num,
                best_frame_image=best.frame_image,
                car_track_id=track_id,
                car_class=self._car_classes.get(track_id, ""),
                total_detections=len(readings),
            ))

        unique_plates.sort(key=lambda p: p.best_frame_num)
        return unique_plates

    @property
    def total_detections(self) -> int:
        return sum(len(r) for r in self._readings.values())
