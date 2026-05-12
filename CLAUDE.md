# RADAR - Real-Time Vehicle Analysis System

## Project Overview
- Streamlit app (`app.py`) for Egyptian license plate detection + OCR (photo & video modes)
- Core ML: YOLOv11m (plate detection), YOLOv11m (char OCR)
- Models live in `models/` and are tracked in git (for team sharing)
- Based on ACLPR project at `/home/mesbah/Desktop/Projects/ACLPR`

## Run
- `source venv/bin/activate && streamlit run app.py`
- Python 3.12, venv (not conda)

## Architecture
- `app.py` — Streamlit GUI (the only entry point, no tkinter/PyQt)
- `core/` — ML modules (model_manager singleton, plate_detector, plate_ocr)
- `pipelines/` — photo_pipeline and video_pipeline orchestration
- `utils/preprocessing.py` — morphological ops, crop helpers
- `config.py` — all thresholds, paths, settings (modified at runtime by Streamlit sidebar)
- `notebooks/` — Kaggle training notebook from ACLPR

## Key Patterns
- `ModelManager` singleton loads models ONCE — never call `YOLO(path)` directly in loops
- `plate_ocr.ocr_yolo()` returns `List[CharDetection]` sorted by x-position (left-to-right spatial order)
- OCR class names are converted from Franco (`noon`, `ain`) to Arabic (`ن`, `ع`) via `config.FRANCO_TO_ARABIC` mapping
- Video pipeline uses `progress_callback(fraction, info_dict)` pattern, not Streamlit-specific progress
- Photo pipeline returns `StepImages` with intermediate BGR arrays for each analysis step

## Future Plans
- Car detection + ByteTrack tracking for video (deduplication, speed estimation)
- Plate aggregator for unique plate identification
- Super-resolution model comparison for plate enhancement
- Seatbelt detection (stub removed, will re-implement later)
- Notebooks folder has plans for training improvements

## User Preferences
- User is a beginner AI developer, based in Egypt, Arabic language context
- Prefers Streamlit over desktop GUI frameworks (tried customtkinter, switched back)
- Plate crop images should NOT be displayed at full container width (they pixelate) — use centered narrow column
- Save uploaded images as PNG (lossless), not JPG (compression degrades OCR)
