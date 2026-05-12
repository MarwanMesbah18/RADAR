from dataclasses import dataclass, field
import numpy as np

from core.plate_utils import separate_chars


def _plates_match(a, b):
    """Check if two UniquePlate results likely belong to the same car."""
    # Same numbers is the strongest signal (most stable part of plate reading)
    if a.numbers and b.numbers and a.numbers == b.numbers:
        return True
    # Same full plate text
    if a.plate_text and a.plate_text == b.plate_text:
        return True
    return False


@dataclass
class PlateReading:
    track_id: int
    frame_num: int
    plate_text: str
    confidence: float
    car_bbox: tuple
    plate_bbox: tuple
    frame_image: np.ndarray = None
    car_crop_image: np.ndarray = None
    plate_crop_image: np.ndarray = None
    ocr_annotated_image: np.ndarray = None
    detections: list = field(default_factory=list)


@dataclass
class UniquePlate:
    plate_text: str
    best_confidence: float
    best_frame_num: int
    best_frame_image: np.ndarray = None
    car_crop_image: np.ndarray = None
    plate_crop_image: np.ndarray = None
    ocr_annotated_image: np.ndarray = None
    car_track_id: int = -1
    car_class: str = ""
    car_bbox: tuple = None
    plate_bbox: tuple = None
    total_detections: int = 0
    numbers: list = field(default_factory=list)
    letters: list = field(default_factory=list)


class PlateAggregator:
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
        # Step 1: pick best reading per track_id
        per_track = []
        for track_id, readings in self._readings.items():
            best = max(readings, key=lambda r: r.confidence)
            chars = separate_chars(best.detections)
            per_track.append(UniquePlate(
                plate_text=chars.text,
                best_confidence=best.confidence,
                best_frame_num=best.frame_num,
                best_frame_image=best.frame_image,
                car_crop_image=best.car_crop_image,
                plate_crop_image=best.plate_crop_image,
                ocr_annotated_image=best.ocr_annotated_image,
                car_track_id=track_id,
                car_class=self._car_classes.get(track_id, ""),
                car_bbox=best.car_bbox,
                plate_bbox=best.plate_bbox,
                total_detections=len(readings),
                numbers=chars.numbers,
                letters=chars.letters,
            ))

        # Step 2: merge duplicate plates from different track IDs
        # (ByteTrack can reassign IDs when it loses a car temporarily)
        merged = []
        for plate in per_track:
            match = None
            for existing in merged:
                if _plates_match(plate, existing):
                    match = existing
                    break
            if match is None:
                merged.append(plate)
            elif plate.best_confidence > match.best_confidence:
                # Replace with the higher-confidence reading
                idx = merged.index(match)
                plate.total_detections += match.total_detections
                merged[idx] = plate
            else:
                match.total_detections += plate.total_detections

        merged.sort(key=lambda p: p.best_frame_num)
        return merged

    @property
    def total_detections(self) -> int:
        return sum(len(r) for r in self._readings.values())
