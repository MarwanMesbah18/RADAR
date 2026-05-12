# Real-Time Software-Based Vehicle & Driver Analysis System

## Project Overview

A real-time computer vision system that ingests camera feeds (USB webcam or RTSP stream), detects vehicles in each frame, and performs four simultaneous analysis tasks on each detected vehicle:

1. **License Plate OCR** — extract and read Egyptian license plates (Arabic letters + Arabic-Indic numerals)
2. **Vehicle Color Classification** — identify the dominant body color
3. **Make & Model Recognition** — classify the vehicle brand and model (e.g., BMW 3 Series)
4. **Seatbelt Detection** — determine if the driver is wearing a seatbelt

**Target performance**: 10-20 FPS on GPU (RTX 3060+), 3-8 FPS on CPU-only.
**Primary language**: Python
**Core frameworks**: Ultralytics YOLOv8, OpenCV, EasyOCR, PyTorch

---

## Developer Profile

- Beginner AI developer with practical YOLO and FaceNet experience
- Solid understanding of core AI concepts (CNNs, object detection, face recognition)
- Based in Egypt — Arabic language context is relevant
- Real-time processing speed is a critical requirement

---

## Architecture Decision: Ensemble Pipeline (Not Multi-Task Model)

This system uses **multiple specialized models running in a coordinated pipeline**, not a single multi-task model.

**Rationale**:
- Each sub-task has fundamentally different input/output shapes (sequence OCR vs binary detection vs multi-class classification)
- Independent failure modes — if OCR fails on a dirty plate, color/make detection still works
- Easier to train, debug, and replace individual components
- You can skip sub-tasks to gain speed on weaker hardware
- Aligns with existing YOLO expertise — start with what you know, add components incrementally

---

## Complete System Data Flow

```
Camera Feed (RTSP / USB Webcam)
        │
        ▼
┌──────────────────────────────────┐
│  Frame Capture & Preprocessing   │
│  - OpenCV VideoCapture           │
│  - Resize to 640x640 for YOLO   │
│  - BGR → RGB conversion          │
│  - Optional: deinterlace, denoise│
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  STAGE 1: Vehicle Detection      │
│  Model: YOLOv8-nano              │
│  Input: 640x640 RGB frame        │
│  Output: List of vehicle bboxes  │
│    [x1, y1, x2, y2, confidence]  │
│  Classes: car, truck, bus, moto  │
│  Latency: ~3ms per frame         │
└──┬───────────┬───────────┬───────┘
   │           │           │
   │     For each detected vehicle bbox:
   │           │           │
   ▼           ▼           ▼
┌─────────────────────────────────────────────────────────┐
│              CROP VEHICLE REGION FROM FRAME             │
│           vehicle_crop = frame[y1:y2, x1:x2]           │
└──┬──────────┬──────────┬──────────┬─────────────────────┘
   │          │          │          │
   ▼          ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌──────────┐
│Sub-P  │ │Sub-P  │ │Sub-P  │ │Sub-P     │
│  1    │ │  2    │ │  3    │ │  4       │
│LP OCR │ │Color  │ │Make/  │ │Seatbelt  │
│       │ │Class  │ │Model  │ │Detect    │
└──┬────┘ └──┬────┘ └──┬────┘ └────┬─────┘
   │         │         │           │
   ▼         ▼         ▼           ▼
┌──────────────────────────────────────────┐
│  Result Aggregation                      │
│  {                                        │
│    "plate": "س ن ١٢٣٤",                  │
│    "color": "white",                      │
│    "make_model": "BMW 3 Series",         │
│    "seatbelt": true,                      │
│    "confidence": { ... },                 │
│    "timestamp": "2026-05-04T14:30:00",   │
│    "vehicle_bbox": [x1,y1,x2,y2]         │
│  }                                        │
└──────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────┐
│  Output Layer                            │
│  - Console print                         │
│  - JSON file logging                     │
│  - Optional: SQLite / REST API           │
│  - Optional: annotated video display     │
└──────────────────────────────────────────┘
```

---

## Sub-Pipeline 1: License Plate OCR (Egyptian/Arabic)

### Data Flow

```
vehicle_crop
    │
    ▼
YOLOv8-nano (fine-tuned for plate detection)
    │  Input: vehicle crop image
    │  Output: plate bounding box [x1,y1,x2,y2]
    │  Confidence threshold: 0.5
    ▼
plate_crop = vehicle_crop[py1:py2, px1:px2]
    │
    ▼
EasyOCR (Arabic mode)
    │  Input: plate crop image
    │  Languages: ['ar']
    │  Output: detected text + confidence per character
    ▼
Post-processing
    │  - Filter low-confidence characters (< 0.4)
    │  - Validate format: Egyptian plates use Arabic letters
    │    + Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩)
    │  - Known format: "letters | number region-code"
    │    e.g., "س ن ١٢٣٤ ب"
    ▼
plate_text: str
```

### Models & Tools

| Component | Model/Tool | Purpose |
|---|---|---|
| Plate detection | YOLOv8-nano (fine-tuned) | Locate plate bounding box within vehicle crop |
| OCR engine | EasyOCR (`lang='ar'`) | Read Arabic characters and numerals from plate crop |

### Training Data for Plate Detection

| Dataset | Description | Link |
|---|---|---|
| ACLPR | Arabic car license plate detection + recognition, supports images and video | [GitHub](https://github.com/AhmAshraf1/ACLPR) |
| Egyptian LP with YOLOv8 | Paper + dataset reference for Egyptian plates specifically | [Springer](https://link.springer.com/article/10.1186/s43067-024-00156-y) |
| Tiny-YOLOv3 Egyptian LP | Full paper with dataset details for Egyptian plates | [PDF](https://thesai.org/Downloads/Volume13No7/Paper_99-Real_time_Egyptian_License_Plate_Detection_and_Recognition.pdf) |
| Custom collection | Capture your own Egyptian plate images with phone camera, label with Roboflow or LabelImg | — |

### Egyptian Plate Format Reference

Egyptian license plates follow a specific format:
- **Top/bottom or left/right split**: Arabic letters on one side, numbers on the other
- **Arabic letters**: ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي
- **Arabic-Indic numerals**: ٠ ١ ٢ ٣ ٤ ٥ ٦ ٧ ٨ ٩ (NOT Western 0-9)
- **Governorate code**: A letter or abbreviation indicating the region (e.g., "ب" for Cairo)

### Key Implementation Notes

- Fine-tune YOLOv8-nano on Egyptian plate images for the plate detection step — do NOT use a general-purpose plate detector
- EasyOCR supports Arabic natively via `reader = easyocr.Reader(['ar'])` — no custom training needed for initial version
- For higher accuracy later: train a custom CRNN (CNN + RNN + CTC loss) on labeled Egyptian plate images
- Plate detection is easier than general object detection because plates have consistent aspect ratio (~4:1 to 5:1) and high contrast

---

## Sub-Pipeline 2: Vehicle Color Classification

### Data Flow

```
vehicle_crop
    │
    ▼
Preprocessing
    │  - Resize to 224x224
    │  - Convert BGR → HSV color space
    │  - Mask out non-body regions (windows, wheels, shadows)
    │    using the vehicle detection bbox inner region
    ▼
Color Extraction (non-DL approach)
    │  - Apply k-means clustering (k=3) on HSV pixel values
    │  - Take the largest cluster as dominant color
    │  - Map HSV centroid to nearest named color via lookup table
    ▼
color_label: str  ("white", "black", "red", "blue", "silver", "gray", "green", "brown", "beige", "yellow")

--- OR (DL approach for higher accuracy) ---

vehicle_crop
    │
    ▼
MobileNetV2 (fine-tuned, last layer replaced with N color classes)
    │  Input: 224x224 RGB
    │  Output: softmax over color classes
    │  Latency: ~2ms per crop
    ▼
color_label: str
```

### Recommended Approach

Start with the **non-DL approach** (HSV + k-means). It covers 90%+ of cases because the vast majority of cars are white, black, silver, gray, red, or blue. Only switch to the DL approach if accuracy is insufficient.

### Color Lookup Table (HSV Ranges)

| Color | H range | S range | V range |
|---|---|---|---|
| White | any | 0-50 | 200-255 |
| Black | any | any | 0-60 |
| Silver/Gray | any | 0-50 | 60-200 |
| Red | 0-10 or 170-180 | 100-255 | 100-255 |
| Blue | 100-130 | 100-255 | 100-255 |
| Green | 40-80 | 100-255 | 100-255 |
| Yellow | 20-40 | 100-255 | 100-255 |
| Brown/Beige | 10-20 | 50-200 | 50-200 |

### Datasets (if using DL approach)

| Dataset | Description |
|---|---|
| [BIT-Vehicle](https://ieeexplore.ieee.org/iel8/6287639/11323511/11363414.pdf) | 9,850 images with color labels |
| [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) | Includes color attribute annotations |

---

## Sub-Pipeline 3: Make & Model Recognition

### Data Flow

```
vehicle_crop
    │
    ▼
Preprocessing
    │  - Resize to 299x299 (InceptionResNetV2 input)
    │  - Normalize with ImageNet mean/std
    │  - Optional: align front grill/headlights as primary features
    ▼
InceptionResNetV2 (fine-tuned on vehicle dataset)
    │  Input: 299x299 RGB
    │  Output: softmax over make+model classes
    │  Latency: ~10ms per crop on GPU
    │  Accuracy: 94.34% (published)
    ▼
Post-processing
    │  - Top-1 prediction as primary label
    │  - Top-3 predictions with confidence scores
    │  - Optional: reject if top-1 confidence < 0.3
    ▼
make_model: str  ("BMW 3 Series", "Mercedes C-Class", etc.)
```

### Models

| Model | Accuracy | Speed | Notes |
|---|---|---|---|
| InceptionResNetV2 | 94.34% | Moderate | Best accuracy, proven on vehicle classification |
| EfficientNet-B3 | ~92% | Faster | Good accuracy/speed tradeoff |
| MobileNetV3-Large | ~88% | Fast | Best for real-time, slightly lower accuracy |
| ResNet-50 | ~90% | Fast | Simple, well-documented baseline |

**Recommendation**: Use **EfficientNet-B3** as the starting point — good accuracy with faster inference than InceptionResNetV2.

### Datasets

| Dataset | Size | Classes | Link |
|---|---|---|---|
| CompCars | 136,726 images | 1,716 car models | [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) |
| Stanford Cars | 16,185 images | 196 classes | [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) |
| BIT-Vehicle | 9,850 images | 6 vehicle types + color | [IEEE](https://ieeexplore.ieee.org/iel8/6287639/11323511/11363414.pdf) |

### Scope Strategy

**Start small**: Train on only the 15-20 most common car makes/models seen on Egyptian roads:
- BMW (3 Series, 5 Series, X3, X5)
- Mercedes (C-Class, E-Class, GLC)
- Toyota (Corolla, Camry, Land Cruiser)
- Hyundai (Accent, Elantra, Tucson)
- Nissan (Sunny, Sentra, Patrol)
- Kia (Sportage, Cerato)
- Chevrolet (Lanor, Optra)

Expand the class list after the initial system is working.

### Key Implementation Notes

- Make/model recognition depends heavily on the **viewing angle** — front 3/4 view and rear 3/4 view give the best results
- The **front grill and headlight shape** are the most discriminative features
- Consider training separate classifiers for front-view vs rear-view if accuracy is insufficient
- [This paper](https://www.mdpi.com/2504-4990/1/2/36) covers real-time VMMR architecture in detail

---

## Sub-Pipeline 4: Seatbelt Detection

### Data Flow

```
vehicle_crop
    │
    ▼
Driver Region Extraction
    │  - Estimate driver position (left side for left-hand drive)
    │  - Crop upper-body region (windshield area)
    │  - driver_crop = vehicle_crop[0:h*0.5, 0:w*0.5]
    ▼
YOLOv8-nano (fine-tuned for seatbelt detection)
    │  Input: driver region crop (640x640 resized)
    │  Output: bounding boxes for seatbelt straps
    │  Classes: "seatbelt" / "no_seatbelt"
    │  Latency: ~3ms per crop
    ▼
Classification Logic
    │  - If seatbelt bbox detected with confidence > 0.5: wearing = True
    │  - Else: wearing = False
    ▼
seatbelt_status: bool
```

### Models

Use **YOLOv8-nano** fine-tuned for 2-class detection (`seatbelt`, `no_seatbelt`).

### Datasets

| Dataset | Size | Format | Link |
|---|---|---|---|
| Seatbelt Detection (Kaggle) | ~2,000 images | 640x640 YOLO format | [Kaggle](https://www.kaggle.com/datasets/alexandresintes/seatbelt-detection-dataset-real-car-photos) |
| Seat Belt Detection (GitHub) | Code + references | YOLOv5/v8/v9 | [GitHub](https://github.com/HayaAbdullahM/Seat-Belt-Detection) |

### Key Challenges

1. **Visibility**: Requires a clear view of the driver's upper body through the windshield. Works best with front-facing cameras at toll gates or intersections.
2. **Lighting**: Night/infrared conditions require a separate model or data augmentation.
3. **Angle**: Overhead cameras (traffic poles) give better visibility than ground-level cameras.
4. **Occlusion**: Sun visors, tinted windows, and reflections can block the seatbelt view.

### Mitigation Strategies

- Apply aggressive data augmentation during training: brightness variation, motion blur, noise
- Use confidence threshold of 0.5 (not too high — seatbelts are thin and easy to miss)
- Mark results as "uncertain" when confidence is between 0.3-0.5

---

## Implementation Phases (Recommended Build Order)

### Phase 1: Core Pipeline (Week 1-2)

**Goal**: Vehicle detection + license plate OCR working in real-time.

```
Tasks:
1. Set up project structure (see below)
2. Implement frame capture loop with OpenCV
3. Integrate YOLOv8-nano for vehicle detection (COCO pre-trained)
4. Add plate detection (fine-tune YOLOv8-nano on Egyptian plate data)
5. Integrate EasyOCR for Arabic plate reading
6. Measure FPS and optimize (resize, skip frames, crop-based inference)
7. Output results to console + JSON
```

**Deliverable**: System reads Egyptian license plates from video in real-time.

### Phase 2: Color + Make/Model (Week 3-4)

**Goal**: Add vehicle color and make/model classification.

```
Tasks:
1. Implement HSV + k-means color classification
2. Collect/curate training data for common Egyptian vehicle makes
3. Fine-tune EfficientNet-B3 on make/model classes
4. Integrate both into the pipeline
5. Run all 3 sub-pipelines in parallel (threading)
6. Re-measure FPS
```

**Deliverable**: System outputs plate + color + make/model.

### Phase 3: Seatbelt + Polish (Week 5-6)

**Goal**: Add seatbelt detection and optimize the full system.

```
Tasks:
1. Download and prepare seatbelt dataset
2. Fine-tune YOLOv8-nano for seatbelt detection
3. Integrate into pipeline
4. Convert all models to ONNX / TensorRT for inference speedup
5. Add annotated video display (bounding boxes + labels on frame)
6. Build JSON logging system
7. End-to-end testing with real camera feeds
```

**Deliverable**: Complete 4-task real-time vehicle analysis system.

---

## Project Structure

```
vehicle-analysis/
├── main.py                      # Entry point: camera loop + pipeline orchestration
├── config.py                    # All config: model paths, thresholds, camera URL
├── requirements.txt             # Python dependencies
├── models/
│   ├── vehicle_detect.pt        # YOLOv8-nano for vehicle detection
│   ├── plate_detect.pt          # YOLOv8-nano fine-tuned for plate detection
│   ├── make_model_cls.pt        # EfficientNet-B3 for make/model
│   └── seatbelt_detect.pt       # YOLOv8-nano fine-tuned for seatbelt
├── pipelines/
│   ├── vehicle_detector.py      # Stage 1: detect vehicles in frame
│   ├── plate_ocr.py             # Sub-pipeline 1: plate detection + EasyOCR
│   ├── color_classifier.py      # Sub-pipeline 2: HSV/k-means or MobileNet
│   ├── make_model.py            # Sub-pipeline 3: make/model classification
│   └── seatbelt_detector.py     # Sub-pipeline 4: seatbelt detection
├── utils/
│   ├── drawing.py               # Annotate frames with results
│   ├── logger.py                # JSON output logging
│   └── preprocessing.py         # Resize, crop, normalize helpers
├── training/
│   ├── train_plate_detector.py  # Fine-tune YOLOv8 on plate data
│   ├── train_make_model.py      # Fine-tune EfficientNet on vehicle data
│   └── train_seatbelt.py        # Fine-tune YOLOv8 on seatbelt data
├── data/
│   ├── plates/                  # Egyptian plate images for training
│   ├── vehicles/                # Vehicle images for make/model training
│   └── seatbelts/               # Seatbelt images for training
└── output/
    ├── logs/                    # JSON result logs
    └── annotated_frames/        # Saved annotated frames (optional)
```

---

## Dependencies (requirements.txt)

```
ultralytics>=8.0.0          # YOLOv8 framework
easyocr>=1.7.0              # Arabic OCR
opencv-python>=4.8.0        # Video capture + image processing
torch>=2.0.0                # PyTorch (model inference)
torchvision>=0.15.0         # Pre-trained models + transforms
numpy>=1.24.0               # Array operations
scikit-learn>=1.3.0         # K-means for color classification
Pillow>=9.0.0               # Image utilities
onnxruntime>=1.16.0         # ONNX inference optimization (Phase 3)
```

Install command:
```bash
pip install ultralytics easyocr opencv-python torch torchvision numpy scikit-learn Pillow onnxruntime
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | None (CPU-only, 3-8 FPS) | NVIDIA RTX 3060 (6GB VRAM, 10-20 FPS) |
| RAM | 8 GB | 16 GB |
| Storage | 5 GB (models + dependencies) | 20 GB (models + training data) |
| Camera | USB webcam 720p | USB webcam 1080p or RTSP stream |

---

## Real-Time Optimization Strategies

### Strategy 1: Crop-Based Inference (Critical)

After Stage 1 (vehicle detection), **only run sub-pipelines on the cropped vehicle region**, never on the full frame. A vehicle crop is typically 200x150 pixels vs the full 1920x1080 frame. This gives a **10-20x speedup** on sub-pipeline models.

### Strategy 2: Frame Skipping

Process every Nth frame instead of every frame:
```python
frame_count = 0
skip_rate = 2  # process every 2nd frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % skip_rate != 0:
        continue  # skip this frame
    # process frame...
```

### Strategy 3: Parallel Sub-Pipelines (Threading)

Run the 4 sub-pipelines concurrently using Python's `concurrent.futures.ThreadPoolExecutor`:
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {
        executor.submit(run_plate_ocr, crop): "plate",
        executor.submit(run_color_classify, crop): "color",
        executor.submit(run_make_model, crop): "make_model",
        executor.submit(run_seatbelt_detect, crop): "seatbelt",
    }
    results = {}
    for future in as_completed(futures):
        key = futures[future]
        results[key] = future.result()
```

### Strategy 4: ONNX / TensorRT Conversion (Phase 3)

Convert PyTorch models to ONNX format for 2-4x inference speedup:
```python
# Export YOLOv8 to ONNX
from ultralytics import YOLO
model = YOLO("plate_detect.pt")
model.export(format="onnx", imgsz=640)
```

### Strategy 5: Async Frame Capture

Use a separate thread for frame capture to avoid blocking the processing loop:
```python
import threading
import queue

frame_queue = queue.Queue(maxsize=2)

def capture_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_queue.full():
            frame_queue.get()  # drop oldest frame
        frame_queue.put(frame)
```

---

## Key Research Papers & References

### License Plate Recognition (Egyptian/Arabic)

| Paper | Year | Key Contribution | Link |
|---|---|---|---|
| Egyptian Car Plate Recognition (YOLOv8 + EasyOCR + CNN) | 2024 | Validates YOLOv8 + EasyOCR pipeline for Egyptian plates | [Springer](https://link.springer.com/article/10.1186/s43067-024-00156-y) |
| Real-time Egyptian LP Detection (Tiny-YOLOv3) | 2022 | Two-model pipeline for Egyptian plates | [ResearchGate](https://www.researchgate.net/publication/362493220_Real-time_Egyptian_License_Plate_Detection_and_Recognition_using_YOLO) |
| Deep Learning for Arabic LP Recognition | 2024 | OCR specifically designed for Arabic characters | [arXiv](https://arxiv.org/pdf/2408.02904) |
| Arabic LP Detection & Recognition (Code) | — | Open-source Arabic plate system, images + video | [GitHub](https://github.com/AhmAshraf1/ACLPR) |

### Vehicle Classification

| Paper | Year | Key Contribution | Link |
|---|---|---|---|
| Ensemble AI for Real-Time Vehicle Recognition | 2024 | 94.78% mAP on BIT-Vehicle dataset | [IEEE](https://ieeexplore.ieee.org/iel8/6287639/11323511/11363414.pdf) |
| Ensemble DL for Vehicle Classification | 2022 | InceptionResNetV2 at 94.34% accuracy | [ResearchGate](https://www.researchgate.net/publication/363773874_Ensemble_Deep_Learning_Models_for_Vehicle_Classification_in_Motorized_Traffic_Analysis) |
| EfficientDet + YOLOv8 Ensemble | 2024 | Vehicle detection + classification ensemble | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11419654/) |
| Real-Time VMMR System | 2024 | Vehicle Make and Model Recognition in real-time | [MDPI](https://www.mdpi.com/2504-4990/1/2/36) |

### Seatbelt Detection

| Paper | Year | Key Contribution | Link |
|---|---|---|---|
| Real-time Seatbelt Detection (YOLO) | 2023 | Comprehensive seatbelt dataset + YOLO detector | [IEEE](https://ieeexplore.ieee.org/document/10063114/) |
| Car Safety Belt Detection (YOLOv7) | 2023 | In-cabin monitoring with YOLOv7 | [MDPI](https://www.mdpi.com/1999-4893/16/9/400) |
| Seat Belt Detection Code | — | YOLOv5/v8/v9 implementations | [GitHub](https://github.com/HayaAbdullahM/Seat-Belt-Detection) |
| Seatbelt Dataset | — | 640x640 labeled images for YOLOv8 | [Kaggle](https://www.kaggle.com/datasets/alexandresintes/seatbelt-detection-dataset-real-car-photos) |

### Vehicle Color

| Dataset | Description | Link |
|---|---|---|
| CompCars | 136K images with color + make/model attributes | [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/) |
| Stanford Cars | 16K images, 196 classes | [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset) |
| BIT-Vehicle | 9,850 images with type + color labels | [IEEE](https://ieeexplore.ieee.org/iel8/6287639/11323511/11363414.pdf) |

---

## Risk Assessment & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Egyptian plate OCR accuracy too low | Medium | High | Fine-tune OCR on collected Egyptian plates; use CRNN+CTC as fallback |
| Seatbelt not visible from camera angle | High | Medium | Mark as "uncertain" rather than false; use front-facing camera position |
| Make/model too many classes | Medium | Low | Start with 15-20 common Egyptian models; expand later |
| Real-time FPS below target | Medium | High | Apply all optimization strategies; drop sub-pipelines selectively |
| GPU not available | Low | Medium | Use YOLOv8-nano + ONNX Runtime on CPU; accept 3-8 FPS |

---

## Testing Strategy

1. **Unit test each sub-pipeline independently** with saved images before integrating
2. **Measure FPS at each integration step** — if adding a sub-pipeline drops FPS below 10, optimize or skip frames
3. **Test with real Egyptian plates** — collect 50+ images from different angles, lighting, and plate conditions
4. **End-to-end test** with a recorded video of cars passing before testing with live camera

---

## Quick Start Commands

```bash
# 1. Create project directory
mkdir -p vehicle-analysis/{models,pipelines,utils,training,data,	output}

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install ultralytics easyocr opencv-python torch torchvision numpy scikit-learn Pillow

# 4. Verify YOLOv8 works
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLOv8 ready')"

# 5. Verify EasyOCR works with Arabic
python -c "import easyocr; reader = easyocr.Reader(['ar']); print('EasyOCR Arabic ready')"

# 6. Download a test video with cars and run vehicle detection
# python main.py --source test_video.mp4
```
