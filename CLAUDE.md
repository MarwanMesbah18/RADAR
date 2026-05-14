# RADAR - Real-Time Vehicle Analysis System

## Project Overview
- Streamlit app (`app.py`) for Egyptian license plate detection + OCR (photo & video modes)
- Core ML: YOLOv11m V1 (plate detection), YOLOv11m V1 + YOLO26m V2 + YOLO26m V2-Weighted-3 (char OCR), YOLO26s (car tracking), YOLOv11m (seatbelt+mobile)
- Models live in `models/` but are gitignored (too large for GitHub — use Git LFS later)
- Based on ACLPR project at `/home/mesbah/Desktop/Projects/ACLPR`

## Run
- `source venv/bin/activate && streamlit run app.py`
- Python 3.12, venv (not conda)

## Architecture

### Entry Point
- `app.py` — Streamlit GUI with sidebar (detection settings, model status) + two tabs (Photo/Video)

### UI Layer (`ui/`)
- `ui/display.py` — shared display functions:
  - `show_image(img_bgr, caption, width, max_height)` — centered image with size constraints
  - `show_enhancement_comparison(plate_crop)` — 3 OCR models x 3 columns (Original, LapSRN, Real-ESRGAN), annotated images only
  - `show_seatbelt_badges(seatbelt_summary)` — colored Streamlit badges for seatbelt/mobile violations
- `ui/photo_tab.py` — Photo mode with two buttons:
  - "Detect Plates Directly" — runs plate detection on full image, shows enhancement+OCR comparison
  - "Detect Cars First" — detects cars as clickable image cards (green/red plate badge), user clicks one, shows plate + comparison
  - "Cars with Plates" filtered section appears below the full car grid
- `ui/video_tab.py` — Video mode with interactive flow:
  - Scan Video → shows car grid as clickable cards → Manual Mode (auto disabled for now) → plate candidate cards → enhancement comparison

### Core ML (`core/`)
- `core/model_manager.py` — singleton loading 7 models: plate detector, 3 OCR models (V1, V2, V2-Weighted-3), car detector, seatbelt detector
- `core/plate_detector.py` — `detect_plates(image)` → `List[PlateDetection]` (sorted widest first)
- `core/plate_ocr.py` — `ocr_yolo(img, model_version=1|2|3)` and `ocr_plate(img, model_version=1|2|3)` — YOLO char detection with `imgsz=640` (no augment — YOLO26 models don't support it)
- `core/car_tracker.py` — `track_cars(frame, persist)` → `List[CarTrack]` using YOLO + ByteTrack
- `core/enhancement.py` — AI super-resolution: `enhance_lapsrn()` and `enhance_realesrgan()`
- `core/plate_utils.py` — `separate_chars()` splits detections into numbers (LTR) and Arabic letters (RTL reversed)
- `core/plate_aggregator.py` — merges duplicate plate readings in video (used by old process_video)
- `core/seatbelt_detector.py` — `detect_seatbelt(image_bgr)` → `List[SeatbeltDetection]` (5 classes: person-noseatbelt, person-seatbelt, seatbelt, windshield, mobile). `get_seatbelt_summary()` returns violation flags. `draw_seatbelt_detections()` annotates with colored boxes

### Pipelines (`pipelines/`)
- `pipelines/photo_pipeline.py` — `analyze_photo()` returns `PhotoAnalysisResult` with vehicles, steps, crops
- `pipelines/video_pipeline.py` — two functions for interactive video:
  - `scan_video_cars(video_path)` — scans all frames, returns `List[ScannedCar]` with dedup (IoU > 0.5 + frame proximity < 30)
  - `find_plate_crops(video_path, scanned_car)` — matches car by spatial proximity (not track ID), returns `List[PlateCandidate]` ranked by crop area
  - `process_video()` — old auto-processing function (kept for future auto mode)

### Config (`config.py`)
- `PLATE_CONFIDENCE = 0.25` — plate detection threshold
- `OCR_CONFIDENCE = 0.55` — OCR character detection threshold
- `CAR_CONFIDENCE = 0.4` — car detection threshold
- `SEATBELT_CONFIDENCE = 0.25` — seatbelt/mobile detection threshold
- `FRANCO_TO_ARABIC` — maps OCR class names (Franco Arabic) to Arabic script

## Key Patterns
- `ModelManager` singleton loads models ONCE — never call `YOLO(path)` directly in loops
- `plate_ocr.ocr_yolo()` returns `List[CharDetection]` sorted by x-position (left-to-right)
- OCR uses `imgsz=640` — MUST match training imgsz or predictions break. No `augment=True` — YOLO26 models don't support test-time augmentation
- `ocr_plate()` takes `model_version`: 1=V1 (YOLOv11m), 2=V2 (YOLO26m), 3=V2 Weighted-3 (YOLO26m)
- Numbers on plates display LTR (no reversal), Arabic letters display RTL (reversed)
- Car selection in both tabs uses clickable image cards with plate/no-plate badges
- Enhancement comparison runs all 3 OCR models side-by-side (V1, V2, V2 Weighted-3), each with 3 columns (Original, LapSRN, Real-ESRGAN). Shows annotated images only (plain image as fallback)
- ByteTrack assigns fresh track IDs on each video pass — NEVER match cars by track_id across passes. Use spatial proximity (IoU on best_bbox) instead
- `find_plate_crops` passes the full `ScannedCar` object, not just track_id
- LapSRN can crash on certain image shapes (cv2.error in merge) — always wrap `_lapsrn_model.upsample()` in try/except with fallback
- Seatbelt detection runs on every car crop (not full image). Results stored as `seatbelt_summary` dict on `VehicleAnalysis` and `ScannedCar`. UI shows colored badges: red "No Belt", green "Belt", red "Phone"
- Seatbelt model: YOLOv11m, `imgsz=640`, trained on merged dataset (v3 base + v2 mobile + v5 unique, 6424 images, 484 mobile labels)

## Next Steps
- **Auto Mode** (disabled) — auto-select best plate crop, run OCR, show results without user picking
- **Improve deduplication** — current IoU-based dedup works but could use visual similarity
- **Improve auto mode** — better "best crop" selection logic (sharpness scoring, not just area)
- **Color classification** — config has `COLOR_KMEANS_CLUSTERS` but not implemented
- **Speed estimation** — using car tracking between frames
- **Notebooks** — has plans for training improvements in `notebooks/`
- YOLO26m OCR model (V2) trained on `characters_final` dataset (9324 images, 98% full plates, ~5 chars/image)
- Ultralytics 8.4.46+ supports `model.model.class_weights = torch.tensor(...)` natively for weighted BCE loss
- Class weighting didn't improve results — normal training with more epochs performed better
- Kaggle T4: batch=16 for YOLOv11m (batch=32 OOMs), batch=32 works for YOLO26m at imgsz=640 (~9GB GPU)
- Seatbelt merged dataset at `Datasets/Seatbelts_Data/seatbelt_merged/` (6424 images, all 640x640, 5 classes). Built from v3 base (640x640, NOT v4 which is 320x320)
- Roboflow YOLO exports sometimes include polygon data in label files (>5 values per line). Truncate to first 5 values (class x y w h) before training

## User Preferences
- User is a beginner AI developer, based in Egypt, Arabic language context
- Prefers Streamlit over desktop GUI frameworks (tried customtkinter, switched back)
- Plate crop images should NOT be displayed at full container width (they pixelate) — use centered narrow column
- Save uploaded images as PNG (lossless), not JPG (compression degrades OCR)
- Uploaded image/video previews should be small (max_height=300 for images, narrow column for video)
- Car and plate selection should be clickable image cards, not dropdown lists
- Auto mode is disabled until "later implementation"
