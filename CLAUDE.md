# RADAR - Real-Time Vehicle Analysis System

## Project Overview
- Streamlit app (`app.py`) for Egyptian license plate detection + OCR (photo & video modes)
- Core ML: YOLOv11m (plate detection), YOLOv11m (char OCR), YOLOv11n + ByteTrack (car tracking)
- Models live in `models/` and are tracked in git (for team sharing)
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
  - `show_enhancement_comparison(plate_crop)` — 3-column comparison (Original, LapSRN, Real-ESRGAN) with annotated OCR images and confidence
- `ui/photo_tab.py` — Photo mode with two buttons:
  - "Detect Plates Directly" — runs plate detection on full image, shows enhancement+OCR comparison
  - "Detect Cars First" — detects cars as clickable image cards (green/red plate badge), user clicks one, shows plate + comparison
- `ui/video_tab.py` — Video mode with interactive flow:
  - Scan Video → shows car grid as clickable cards → Manual Mode (auto disabled for now) → plate candidate cards → enhancement comparison

### Core ML (`core/`)
- `core/model_manager.py` — singleton that loads YOLO models once (plate detector, plate OCR, car detector)
- `core/plate_detector.py` — `detect_plates(image)` → `List[PlateDetection]` (sorted widest first)
- `core/plate_ocr.py` — `ocr_yolo()` and `ocr_plate()` — YOLO char detection with `augment=True, imgsz=1280`
- `core/car_tracker.py` — `track_cars(frame, persist)` → `List[CarTrack]` using YOLO + ByteTrack
- `core/enhancement.py` — AI super-resolution: `enhance_lapsrn()` and `enhance_realesrgan()` (no basic method)
- `core/plate_utils.py` — `separate_chars()` splits detections into numbers (LTR) and Arabic letters (RTL reversed)
- `core/plate_aggregator.py` — merges duplicate plate readings in video (used by old process_video)

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
- `FRANCO_TO_ARABIC` — maps OCR class names (Franco Arabic) to Arabic script

## Key Patterns
- `ModelManager` singleton loads models ONCE — never call `YOLO(path)` directly in loops
- `plate_ocr.ocr_yolo()` returns `List[CharDetection]` sorted by x-position (left-to-right)
- OCR uses `augment=True` (test-time augmentation) and `imgsz=1280` for better accuracy
- Numbers on plates display LTR (no reversal), Arabic letters display RTL (reversed)
- Car selection in both tabs uses clickable image cards with plate/no-plate badges
- Enhancement comparison shows Original + LapSRN + Real-ESRGAN with annotated OCR and confidence
- Video pipeline matches cars by spatial proximity (IoU on best_bbox), not ByteTrack track IDs
- `find_plate_crops` passes the full `ScannedCar` object, not just track_id

## Next Steps
- **Auto Mode** (disabled) — auto-select best plate crop, run OCR, show results without user picking
- **Improve deduplication** — current IoU-based dedup works but could use visual similarity
- **Improve auto mode** — better "best crop" selection logic (sharpness scoring, not just area)
- **Seatbelt detection** — will re-implement later
- **Color classification** — config has `COLOR_KMEANS_CLUSTERS` but not implemented
- **Speed estimation** — using car tracking between frames
- **Notebooks** — has plans for training improvements in `notebooks/`

## User Preferences
- User is a beginner AI developer, based in Egypt, Arabic language context
- Prefers Streamlit over desktop GUI frameworks (tried customtkinter, switched back)
- Plate crop images should NOT be displayed at full container width (they pixelate) — use centered narrow column
- Save uploaded images as PNG (lossless), not JPG (compression degrades OCR)
- Uploaded image/video previews should be small (max_height=300 for images, narrow column for video)
- Car and plate selection should be clickable image cards, not dropdown lists
- Auto mode is disabled until "later implementation"
