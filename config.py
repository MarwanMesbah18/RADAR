import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11m_car_plate_trained.pt")
PLATE_OCR_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11m_car_plate_ocr.pt")

# Detection thresholds
PLATE_CONFIDENCE = 0.25
OCR_CONFIDENCE = 0.25
OCR_CHAR_MIN_CONFIDENCE = 0.4
VIDEO_OCR_MIN_CONFIDENCE = 0.75  # Only store readings >= 75% confidence in video

# Car detection (YOLOv11n COCO pretrained)
CAR_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11n.pt")
CAR_CONFIDENCE = 0.4
CAR_CLASSES = [2, 5, 7]  # car, bus, truck
TRACK_BUFFER = 30  # ByteTrack track persistence (frames)

# Franco → Arabic character mapping (OCR model uses transliterated class names)
FRANCO_TO_ARABIC = {
    'alif': 'ا', 'baa': 'ب', 'jeem': 'ج', 'daal': 'د',
    'haa': 'ھ', '7aa': 'ح', 'waw': 'و', 'zaal': 'ذ',
    'raa': 'ر', 'zay': 'ز', 'seen': 'س', 'sheen': 'ش',
    'saad': 'ص', 'daad': 'ض', 'taa': 'ت', 'Taa': 'ط',
    'thaa': 'ث', 'Thaa': 'ث', 'ain': 'ع', 'ghayn': 'غ',
    'faa': 'ف', 'qaaf': 'ق', 'kaaf': 'ك', 'laam': 'ل',
    'meem': 'م', 'noon': 'ن', 'khaa': 'خ', 'yaa': 'ي',
}

# Preprocessing
MORPH_KERNEL_SIZE = (1, 1)
BINARY_THRESHOLD = 128
PLATE_CROP_MARGIN = 20

# Color classification
COLOR_KMEANS_CLUSTERS = 3

# GUI
WINDOW_TITLE = "RADAR - Vehicle Analysis System"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Output
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
