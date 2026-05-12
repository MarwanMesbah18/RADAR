from ultralytics import YOLO
import config


class ModelManager:
    _instance = None

    def __init__(self):
        self._plate_detector = None
        self._plate_ocr = None
        self._car_detector = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_plate_detector(self):
        if self._plate_detector is None:
            self._plate_detector = YOLO(config.PLATE_MODEL_PATH)
        return self._plate_detector

    def get_plate_ocr(self):
        if self._plate_ocr is None:
            self._plate_ocr = YOLO(config.PLATE_OCR_MODEL_PATH)
        return self._plate_ocr

    def get_car_detector(self):
        if self._car_detector is None:
            self._car_detector = YOLO(config.CAR_MODEL_PATH)
        return self._car_detector
