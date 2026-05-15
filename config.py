import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLATE_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11m_car_plate_trained_V1.pt")
PLATE_OCR_MODEL_PATH = os.path.join(MODELS_DIR, "yolo11m_car_plate_ocr_V1.pt")
PLATE_OCR_V2_MODEL_PATH = os.path.join(MODELS_DIR, "yolo26m_car_plate_ocr_V2.pt")
PLATE_OCR_V2W_MODEL_PATH = os.path.join(MODELS_DIR, "yolo26m_car_plate_ocr_V2_Weighted-3.pt")
CAR_MODEL_PATH = os.path.join(MODELS_DIR, "yolo26s_cars.pt")
SEATBELT_MODEL_PATH = os.path.join(MODELS_DIR, "seatbelt_mobile_v1.pt")

# Detection thresholds (user-adjustable via sidebar)
PLATE_CONFIDENCE = 0.25
OCR_CONFIDENCE = 0.55
CAR_CONFIDENCE = 0.4
SEATBELT_CONFIDENCE = 0.25

# Cache-min thresholds (run models at this conf, then filter client-side)
PLATE_CONFIDENCE_CACHE = 0.01
OCR_CONFIDENCE_CACHE = 0.01
CAR_CONFIDENCE_CACHE = 0.01
SEATBELT_CONFIDENCE_CACHE = 0.01

# Car detection classes
CAR_CLASSES = [2, 5, 7]  # car, bus, truck

# OCR
OCR_CHAR_MIN_CONFIDENCE = 0.4

# Franco → Arabic character mapping (OCR model uses transliterated class names)
FRANCO_TO_ARABIC = {
    'alif': 'ا', 'baa': 'ب', 'jeem': 'ج', 'daal': 'د',
    'haa': 'ھ', '7aa': 'ح', 'waw': 'و', 'zaal': 'ذ',
    'raa': 'ر', 'zay': 'ز', 'seen': 'س', 'sheen': 'ش',
    'saad': 'ص', 'daad': 'ض', 'taa': 'ت', 'Taa': 'ط',
    'thaa': 'ث', 'Thaa': 'ظ', 'ain': 'ع', 'ghayn': 'غ',
    'faa': 'ف', 'qaaf': 'ق', 'kaaf': 'ك', 'laam': 'ل',
    'meem': 'م', 'noon': 'ن', 'khaa': 'خ', 'yaa': 'ي',
}

# Preprocessing
PLATE_CROP_MARGIN = 20
