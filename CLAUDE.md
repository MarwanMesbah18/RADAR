# RADAR - Real-Time Vehicle Analysis System

## Project Overview
- Streamlit app (`app.py`) for Egyptian license plate detection + OCR (photo mode only)
- Core ML: YOLOv11m V1 (plate detection), YOLOv11m V1 + YOLO26m V2 + YOLO26m V2-Weighted-3 (char OCR), YOLO26s (car detection), YOLOv11m (seatbelt+mobile)
- Models live in `models/` but are gitignored (too large for GitHub — use Git LFS later)
- Based on ACLPR project at `/home/mesbah/Desktop/Projects/ACLPR`

## Run
- `python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
- `streamlit run app.py`
- Python 3.12, venv (not conda)

## Architecture

### Entry Point
- `app.py` — Streamlit sidebar (detection sliders, model status) + photo mode only (no tabs)

### UI Layer (`ui/`)
- `ui/display.py` — shared display functions:
  - `show_image(img_bgr, caption, width, max_height)` — centered image with size constraints
  - `show_enhancement_comparison(plate_crop)` — 3 OCR models x 3 columns (Original, LapSRN, Real-ESRGAN). Caches enhancement images + raw OCR results in session_state for live filtering
  - `show_seatbelt_badges(seatbelt_summary)` — colored badges for violations
  - `_draw_ocr_annotations(base_img, detections)` — per-class colored boxes with Franco class names (not Arabic)
- `ui/photo_tab.py` — Photo mode:
  - Sample images gallery (expandable, clickable thumbnails)
  - Upload or select image → detect cars → clickable car grid → interior analysis + plate OCR

### Core ML (`core/`)
- `core/model_manager.py` — singleton loading models: plate detector, 3 OCR models, car detector, seatbelt detector
- `core/plate_detector.py` — `detect_plates(image, conf=None)` → `List[PlateDetection]` (sorted widest first). Optional `conf` param overrides `config.PLATE_CONFIDENCE`
- `core/plate_ocr.py` — `ocr_yolo(img, model_version=1|2|3, conf=None)` and `ocr_plate(img, model_version=1|2|3, conf=None)`. Optional `conf` param for cache-min runs
- `core/car_tracker.py` — `track_cars(frame, persist, conf=None)` → `List[CarTrack]` using YOLO + ByteTrack
- `core/enhancement.py` — AI super-resolution: `enhance_lapsrn()` and `enhance_realesrgan()`
- `core/plate_utils.py` — `separate_chars()` splits detections into numbers (LTR) and Arabic letters (RTL reversed)
- `core/seatbelt_detector.py` — `detect_seatbelt(image_bgr, conf=None)` → `List[SeatbeltDetection]`. `draw_seatbelt_detections()` uses font scale 0.7, thickness 2

### Pipelines (`pipelines/`)
- `pipelines/photo_pipeline.py` — `detect_cars_step(bgr_image, conf=None)` and `analyze_plates_step(...)` for step-by-step photo analysis

### Utils (`utils/`)
- `utils/preprocessing.py` — `pil_to_cv2()`, `put_arabic_text()`, `crop_to_bbox()`, `ensure_valid_bbox()`

### Config (`config.py`)
- Detection thresholds: `PLATE_CONFIDENCE=0.25`, `OCR_CONFIDENCE=0.55`, `CAR_CONFIDENCE=0.4`, `SEATBELT_CONFIDENCE=0.25`
- Cache-min thresholds: `PLATE_CONFIDENCE_CACHE=0.01` etc. — used for Roboflow-style live filtering
- `FRANCO_TO_ARABIC` — maps OCR class names (Franco) to Arabic script

## Key Patterns

### Roboflow-Style Live Filtering
- All models run ONCE at cache-min confidence (0.01), store raw results in `st.session_state`
- Sidebar sliders filter cached results client-side — no model re-run on slider change
- Photo tab: car detection, seatbelt, plate detection, OCR all cached at min conf
- `show_enhancement_comparison()` caches enhanced images + OCR results, filters by current slider

### OCR Annotations
- `CharDetection.class_name` contains Arabic text (converted by `_to_arabic()`)
- Annotation labels use Franco names via `_ARABIC_TO_FRANCO` reverse mapping
- Per-class colors via `_char_color(class_name)` using hash-based palette
- When no filtering applied, uses YOLO's `result[0].plot()` directly (original colors)

### Streamlit Gotchas
- `on_change` callbacks run BEFORE the script — never clear widget keys (like `photo_upload`) in callbacks or the widget resets
- `tempfile.NamedTemporaryFile` for uploads — cache path in session_state, write ONCE, don't recreate every rerun
- When car grid count changes (filtered by slider), invalidate `selected_car` index since position shifts
- Sliders use `step=0.01` for fine-grained control — user can click and type exact values

### Model Usage
- `ModelManager` singleton loads models ONCE — never call `YOLO(path)` directly
- All core functions accept optional `conf` parameter to override threshold
- OCR uses `imgsz=640` — MUST match training. No `augment=True` (YOLO26 doesn't support it)
- LapSRN can crash on certain shapes — always wrap in try/except with fallback
- Numbers display LTR, Arabic letters display RTL (reversed by `separate_chars()`)

## Next Steps
- Video mode was removed — tracking was unreliable, plate-first approach too slow. Revisit later with better approach
- Color classification — config has `COLOR_KMEANS_CLUSTERS` but not implemented
- Speed estimation — using car tracking between frames
- Notebooks — training improvements in `notebooks/`
- Kaggle T4: batch=16 for YOLOv11m (batch=32 OOMs), batch=32 for YOLO26m at imgsz=640
- Roboflow YOLO exports sometimes include polygon data (>5 values per line). Truncate to first 5

## User Preferences
- Beginner AI developer, based in Egypt, Arabic language context
- Prefers Streamlit over desktop GUI frameworks (tried customtkinter, switched back)
- Plate crop images should NOT be displayed at full container width — use centered narrow column
- Save uploaded images as PNG (lossless), not JPG (compression degrades OCR)
- Car and plate selection should be clickable image cards, not dropdown lists
- Sample images: compact gallery grid, no text, expandable section
- Sidebar sliders should allow fine-grained values (step=0.01), user wants to type exact numbers
