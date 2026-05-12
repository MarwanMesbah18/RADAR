# Unified Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign photo and video tabs to use an interactive flow ending with a shared 3-column (Original/LapSRN/Real-ESRGAN) enhancement+OCR comparison.

**Architecture:** Add a shared `show_enhancement_comparison()` display function, rewrite `ui/photo_tab.py` with two entry buttons (direct plate / cars first), rewrite `ui/video_tab.py` with scan→select→auto/manual→comparison flow, and add two new pipeline functions (`scan_video_cars`, `find_plate_crops`) to `pipelines/video_pipeline.py`.

**Tech Stack:** Streamlit, OpenCV, YOLO (ultralytics), NumPy, PIL

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `ui/display.py` | Modify | Add shared `show_enhancement_comparison()` |
| `ui/photo_tab.py` | Rewrite | Two-button flow: direct plate OR cars→plates→comparison |
| `ui/video_tab.py` | Rewrite | Scan→select car→auto/manual→comparison |
| `pipelines/video_pipeline.py` | Modify | Add `scan_video_cars()` and `find_plate_crops()` + new data classes |
| `pipelines/photo_pipeline.py` | No changes | Photo pipeline already returns everything we need |

---

### Task 1: Add shared enhancement comparison to `ui/display.py`

**Files:**
- Modify: `ui/display.py`

This adds the shared 3-column comparison function used by both tabs.

- [ ] **Step 1: Add imports and `show_enhancement_comparison` function to `ui/display.py`**

Append after the existing `show_vehicle_analysis` function:

```python
from core.enhancement import enhance_lapsrn, enhance_realesrgan
from core.plate_ocr import ocr_plate


def show_enhancement_comparison(plate_crop):
    """Show 3-column enhancement + OCR comparison: Original, LapSRN, Real-ESRGAN.

    Each column shows the image on top, OCR text + confidence below.
    plate_crop: BGR numpy array of the cropped plate.
    """
    col_orig, col_lap, col_esrgan = st.columns(3)

    # Run enhancements
    lapsrn_img = enhance_lapsrn(plate_crop)
    esrgan_img = enhance_realesrgan(plate_crop)

    # Run OCR on all three
    ocr_orig = ocr_plate(plate_crop)
    ocr_lap = ocr_plate(lapsrn_img)
    ocr_esrgan = ocr_plate(esrgan_img)

    with col_orig:
        st.markdown("**Original**")
        st.image(plate_crop, channels="BGR")
        st.markdown(f"Plate: `{ocr_orig.text or '—'}`")
        st.markdown(f"Confidence: `{ocr_orig.confidence:.0%}`")

    with col_lap:
        st.markdown("**LapSRN (AI Light)**")
        st.image(lapsrn_img, channels="BGR")
        st.markdown(f"Plate: `{ocr_lap.text or '—'}`")
        st.markdown(f"Confidence: `{ocr_lap.confidence:.0%}`")

    with col_esrgan:
        st.markdown("**Real-ESRGAN (AI Heavy)**")
        st.image(esrgan_img, channels="BGR")
        st.markdown(f"Plate: `{ocr_esrgan.text or '—'}`")
        st.markdown(f"Confidence: `{ocr_esrgan.confidence:.0%}`")
```

- [ ] **Step 2: Verify display.py loads without errors**

Run: `cd /home/mesbah/Desktop/Projects/RADAR && source venv/bin/activate && python -c "import ast; ast.parse(open('ui/display.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ui/display.py
git commit -m "feat: add shared enhancement+OCR comparison display function"
```

---

### Task 2: Rewrite `ui/photo_tab.py` with two-button flow

**Files:**
- Rewrite: `ui/photo_tab.py`

This replaces the current step-by-step flow with two buttons: "Detect Plates Directly" and "Detect Cars First". Both paths end with the shared `show_enhancement_comparison()`.

- [ ] **Step 1: Rewrite `ui/photo_tab.py`**

Replace the entire file with:

```python
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile

from pipelines.photo_pipeline import analyze_photo
from core.plate_detector import detect_plates
from utils.preprocessing import pil_to_cv2
from ui.display import show_image, show_enhancement_comparison


def _reset():
    for key in ["photo_mode", "photo_result", "photo_direct_plates",
                "selected_car", "tmp_file"]:
        st.session_state.pop(key, None)


def render_photo_tab():
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"],
        key="photo_upload", on_change=_reset,
    )

    if uploaded is None:
        return

    if "tmp_file" not in st.session_state:
        image = Image.open(uploaded).convert("RGB")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(tmp.name, format="PNG")
        tmp.close()
        st.session_state["tmp_file"] = tmp.name

    tmp_path = st.session_state["tmp_file"]
    img = cv2.imread(tmp_path)
    if img is not None:
        show_image(img, "Original Image", width=5)

    st.markdown("---")

    # Two mode buttons
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        btn_direct = st.button("🔍 Detect Plates Directly", type="primary",
                               use_container_width=True)
    with col_b2:
        btn_cars = st.button("🚗 Detect Cars First", type="primary",
                             use_container_width=True)

    if btn_direct:
        _reset()
        st.session_state["photo_mode"] = "direct"
    elif btn_cars:
        _reset()
        st.session_state["photo_mode"] = "cars"

    mode = st.session_state.get("photo_mode")
    if mode == "direct":
        _render_direct_mode(tmp_path)
    elif mode == "cars":
        _render_cars_mode(tmp_path)


def _render_direct_mode(tmp_path):
    """Detect plates directly on the full image."""
    if "photo_direct_plates" not in st.session_state:
        with st.spinner("Detecting plates..."):
            pil_img = Image.open(tmp_path).convert("RGB")
            plates = detect_plates(pil_img)
            st.session_state["photo_direct_plates"] = plates

    plates = st.session_state["photo_direct_plates"]

    if not plates:
        st.warning("No plates detected on this image.")
        return

    st.success(f"Found {len(plates)} plate(s)")
    st.markdown("---")

    for i, plate in enumerate(plates):
        st.markdown(f"#### Plate #{i + 1} (confidence: {plate.confidence:.0%})")
        plate_bgr = pil_to_cv2(plate.cropped_image)
        show_enhancement_comparison(plate_bgr)
        if i < len(plates) - 1:
            st.markdown("---")


def _render_cars_mode(tmp_path):
    """Detect cars first, let user select, then detect plate."""
    if "photo_result" not in st.session_state:
        with st.spinner("Detecting cars..."):
            st.session_state["photo_result"] = analyze_photo(tmp_path, multi_car=True)

    result = st.session_state["photo_result"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Cars Found", result.cars_found)
    col2.metric("Plates Found", result.plates_found)
    col3.metric("Time", f"{result.processing_time:.2f}s")

    if result.cars_found == 0:
        st.warning("No vehicles detected.")
        return

    if result.all_cars_image is not None:
        show_image(result.all_cars_image, "All detected vehicles", width=5)

    st.markdown("---")

    # Car selection
    car_options = {}
    cars_with_plates = [(i, v) for i, v in enumerate(result.vehicles) if v.plate_detection is not None]
    cars_without = [(i, v) for i, v in enumerate(result.vehicles) if v.plate_detection is None]

    for i, v in cars_with_plates:
        car_options[f"Car #{i + 1} (plate found)"] = i
    for i, v in cars_without:
        car_options[f"Car #{i + 1} (no plate)"] = i

    if not car_options:
        st.warning("No cars detected.")
        return

    selected_label = st.selectbox("Choose a car:", list(car_options.keys()), index=0)
    st.session_state["selected_car"] = car_options[selected_label]

    if not st.button("🔢 Show Plate & OCR", type="primary", use_container_width=True):
        return

    idx = st.session_state.get("selected_car", 0)
    v = result.vehicles[idx]

    # Show car crop
    if v.steps and v.steps.car_crop is not None:
        st.markdown("**Vehicle Crop**")
        st.image(v.steps.car_crop, channels="BGR")

    if v.plate_detection is None:
        st.info("No plate detected on this car.")
        return

    # Show plate crop
    if v.steps and v.steps.plate_crop is not None:
        st.markdown("**Plate Crop**")
        st.image(v.steps.plate_crop, channels="BGR")

    st.markdown("---")
    st.markdown("#### Enhancement & OCR Comparison")

    plate_crop = v.steps.plate_crop
    if plate_crop is None:
        plate_crop = pil_to_cv2(v.plate_detection.cropped_image)

    show_enhancement_comparison(plate_crop)
```

- [ ] **Step 2: Verify photo_tab.py loads without errors**

Run: `cd /home/mesbah/Desktop/Projects/RADAR && source venv/bin/activate && python -c "import ast; ast.parse(open('ui/photo_tab.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ui/photo_tab.py
git commit -m "feat: rewrite photo tab with two-button flow and shared comparison"
```

---

### Task 3: Add `scan_video_cars` and `find_plate_crops` to video pipeline

**Files:**
- Modify: `pipelines/video_pipeline.py`

Add new data classes and two new pipeline functions that support the interactive video flow. The existing `process_video()` function is kept unchanged.

- [ ] **Step 1: Add new data classes and functions to `pipelines/video_pipeline.py`**

Append the following at the end of the file (after the existing `process_video` function):

```python
# ── Interactive video pipeline functions ──


@dataclass
class ScannedCar:
    track_id: int
    car_class: str
    best_crop: object = None  # numpy array
    best_frame_num: int = 0
    first_seen_frame: int = 0


@dataclass
class PlateCandidate:
    frame_num: int
    plate_crop: object = None  # numpy array (BGR)
    plate_bbox: tuple = None   # (x1, y1, x2, y2) in frame coords
    car_crop: object = None    # numpy array (BGR)
    crop_area: float = 0.0


def scan_video_cars(video_path, progress_callback=None):
    """Scan a video to find all unique cars via ByteTrack tracking.

    Returns list of ScannedCar sorted by first_seen_frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    all_cars: dict[int, ScannedCar] = {}
    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        try:
            car_tracks = track_cars(frame, persist=True)

            for car in car_tracks:
                car_crop = _crop_car(frame, car.bbox)
                if car_crop.size == 0:
                    continue

                if car.track_id not in all_cars:
                    all_cars[car.track_id] = ScannedCar(
                        track_id=car.track_id,
                        car_class=car.class_name,
                        first_seen_frame=current_frame,
                    )

                car_obj = all_cars[car.track_id]
                curr_area = car_crop.shape[0] * car_crop.shape[1]
                prev_area = car_obj.best_crop.shape[0] * car_obj.best_crop.shape[1] if car_obj.best_crop is not None else 0
                if curr_area > prev_area:
                    car_obj.best_crop = car_crop.copy()
                    car_obj.best_frame_num = current_frame
        except Exception as e:
            print(f"Error scanning frame {current_frame}: {e}")

        if progress_callback:
            progress_callback(current_frame / max(total_frames, 1), {
                "frame": current_frame,
                "total": total_frames,
                "cars": len(all_cars),
            })

    cap.release()

    cars = sorted(all_cars.values(), key=lambda c: c.first_seen_frame)
    return cars


def find_plate_crops(video_path, target_track_id, top_n=5):
    """Find plate crop candidates for a specific tracked car in a video.

    Seeks through all frames, runs plate detection only on frames where the
    target car appears, returns top_n candidates sorted by crop area (largest first).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    candidates: list[PlateCandidate] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        current_frame += 1

        try:
            car_tracks = track_cars(frame, persist=True)

            for car in car_tracks:
                if car.track_id != target_track_id:
                    continue

                car_crop = _crop_car(frame, car.bbox)
                if car_crop.size == 0:
                    continue

                cx1, cy1 = int(car.bbox[0]), int(car.bbox[1])
                car_pil = Image.fromarray(cv2.cvtColor(car_crop, cv2.COLOR_BGR2RGB))
                plate_dets = detect_plates(car_pil)
                if not plate_dets:
                    continue

                plate = plate_dets[0]
                px1, py1, px2, py2 = [int(v) for v in plate.bbox]

                # Plate crop in frame coordinates
                fx1, fy1 = cx1 + px1, cy1 + py1
                fx2, fy2 = cx1 + px2, cy1 + py2
                fx1, fy1, fx2, fy2 = ensure_valid_bbox(
                    (fx1, fy1, fx2, fy2), frame_width, frame_height
                )

                plate_crop = frame[fy1:fy2, fx1:fx2]
                if plate_crop.size == 0:
                    continue

                crop_area = (fx2 - fx1) * (fy2 - fy1)

                # Skip if very similar frame already captured (deduplicate by area proximity)
                is_dup = any(abs(c.crop_area - crop_area) / max(crop_area, 1) < 0.1 for c in candidates)
                if is_dup:
                    continue

                candidates.append(PlateCandidate(
                    frame_num=current_frame,
                    plate_crop=plate_crop.copy(),
                    plate_bbox=(fx1, fy1, fx2, fy2),
                    car_crop=car_crop.copy(),
                    crop_area=crop_area,
                ))
        except Exception as e:
            print(f"Error finding plates at frame {current_frame}: {e}")

    cap.release()

    # Sort by crop area descending, take top_n
    candidates.sort(key=lambda c: c.crop_area, reverse=True)
    return candidates[:top_n]
```

- [ ] **Step 2: Verify video_pipeline.py loads without errors**

Run: `cd /home/mesbah/Desktop/Projects/RADAR && source venv/bin/activate && python -c "import ast; ast.parse(open('pipelines/video_pipeline.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pipelines/video_pipeline.py
git commit -m "feat: add scan_video_cars and find_plate_crops for interactive video flow"
```

---

### Task 4: Rewrite `ui/video_tab.py` with interactive flow

**Files:**
- Rewrite: `ui/video_tab.py`

This replaces the automatic video processing with an interactive flow: scan → select car → auto/manual → enhancement comparison.

- [ ] **Step 1: Rewrite `ui/video_tab.py`**

Replace the entire file with:

```python
import streamlit as st
import tempfile

from pipelines.video_pipeline import scan_video_cars, find_plate_crops
from ui.display import show_enhancement_comparison


def _reset_scan():
    for key in ["video_cars", "video_selected_car", "video_plate_mode",
                "video_plate_crop", "video_candidates"]:
        st.session_state.pop(key, None)


def render_video_tab():
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"],
        key="video_upload", on_change=_reset_scan,
    )

    if uploaded_video is None:
        return

    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())
    tfile.close()

    with st.expander("Original Video", expanded=False):
        st.video(tfile.name)

    st.markdown("---")

    # ── Step 1: Scan Video ──
    if not st.button("🎬 Scan Video", type="primary", use_container_width=True):
        # If already scanned, show results
        if "video_cars" in st.session_state:
            _render_car_selection(tfile.name)
        return

    _reset_scan()
    progress_bar = st.progress(0, text="Scanning video for cars...")

    def on_progress(fraction, info):
        progress_bar.progress(
            fraction,
            text=f"Frame {info['frame']}/{info['total']} — {info['cars']} cars found",
        )

    with st.spinner("Scanning video..."):
        cars = scan_video_cars(tfile.name, progress_callback=on_progress)

    progress_bar.progress(1.0, text=f"Done! Found {len(cars)} unique cars.")
    st.session_state["video_cars"] = cars

    if not cars:
        st.warning("No cars detected in this video.")
        return

    _render_car_selection(tfile.name)


def _render_car_selection(video_path):
    """Show car grid and let user pick one."""
    cars = st.session_state["video_cars"]

    st.markdown("### Detected Vehicles")
    st.markdown(f"Found **{len(cars)}** unique vehicles in this video.")

    # Car grid: 4 per row
    cols_per_row = 4
    car_labels = []
    for i in range(0, len(cars), cols_per_row):
        row = st.columns(cols_per_row)
        for j, car in enumerate(cars[i:i + cols_per_row]):
            with row[j]:
                if car.best_crop is not None:
                    st.image(car.best_crop, channels="BGR")
                label = f"Car #{car.track_id} ({car.car_class})"
                st.caption(label)
                car_labels.append(label)

    st.markdown("---")

    selected = st.selectbox("Select a car to analyze:", car_labels, index=0)
    selected_idx = car_labels.index(selected)
    st.session_state["video_selected_car"] = selected_idx

    # Mode buttons
    col_auto, col_manual = st.columns(2)
    with col_auto:
        btn_auto = st.button("⚡ Auto Mode", type="primary", use_container_width=True)
    with col_manual:
        btn_manual = st.button("🎯 Manual Mode", type="primary", use_container_width=True)

    if btn_auto:
        st.session_state["video_plate_mode"] = "auto"
    elif btn_manual:
        st.session_state["video_plate_mode"] = "manual"

    mode = st.session_state.get("video_plate_mode")
    if mode == "auto":
        _render_auto_mode(video_path, cars[selected_idx])
    elif mode == "manual":
        _render_manual_mode(video_path, cars[selected_idx])


def _render_auto_mode(video_path, car):
    """Auto: find the best plate crop for this car."""
    st.markdown("### Auto Mode — Finding best plate...")

    with st.spinner("Scanning frames for plate..."):
        candidates = find_plate_crops(video_path, car.track_id, top_n=1)

    if not candidates:
        st.warning(f"No plate detected for Car #{car.track_id}.")
        return

    best = candidates[0]
    st.success(f"Plate found at frame {best.frame_num} (crop area: {best.crop_area:.0f} px²)")

    col_car, col_plate = st.columns(2)
    with col_car:
        st.markdown("**Vehicle**")
        st.image(best.car_crop, channels="BGR")
    with col_plate:
        st.markdown("**Plate**")
        st.image(best.plate_crop, channels="BGR")

    st.markdown("---")
    st.markdown("### Enhancement & OCR Comparison")
    show_enhancement_comparison(best.plate_crop)


def _render_manual_mode(video_path, car):
    """Manual: show multiple plate candidates, let user pick."""
    st.markdown("### Manual Mode — Select best plate")

    if "video_candidates" not in st.session_state:
        with st.spinner("Scanning frames for plate candidates..."):
            candidates = find_plate_crops(video_path, car.track_id, top_n=5)
            st.session_state["video_candidates"] = candidates

    candidates = st.session_state["video_candidates"]

    if not candidates:
        st.warning(f"No plate detected for Car #{car.track_id}.")
        return

    # Show candidates
    labels = [
        f"Frame {c.frame_num} (area: {c.crop_area:.0f} px²)"
        for c in candidates
    ]

    selected_label = st.selectbox("Choose a plate candidate:", labels, index=0)
    selected_idx = labels.index(selected_label)
    chosen = candidates[selected_idx]

    col_car, col_plate = st.columns(2)
    with col_car:
        st.markdown("**Vehicle**")
        st.image(chosen.car_crop, channels="BGR")
    with col_plate:
        st.markdown("**Plate**")
        st.image(chosen.plate_crop, channels="BGR")

    st.markdown("---")
    st.markdown("### Enhancement & OCR Comparison")
    show_enhancement_comparison(chosen.plate_crop)
```

- [ ] **Step 2: Verify video_tab.py loads without errors**

Run: `cd /home/mesbah/Desktop/Projects/RADAR && source venv/bin/activate && python -c "import ast; ast.parse(open('ui/video_tab.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add ui/video_tab.py
git commit -m "feat: rewrite video tab with interactive scan-select-compare flow"
```

---

### Task 5: Smoke test the full app

- [ ] **Step 1: Start the Streamlit app**

Run: `cd /home/mesbah/Desktop/Projects/RADAR && source venv/bin/activate && streamlit run app.py --server.headless true &`

- [ ] **Step 2: Verify no import errors in terminal**

Check terminal output for any `ImportError` or `ModuleNotFoundError`. The app should start without errors.

- [ ] **Step 3: Commit any fixes if needed**

If any import or startup errors were found and fixed:

```bash
git add -A
git commit -m "fix: resolve import/startup issues from unified pipeline rewrite"
```

---

## Verification Checklist

After all tasks are complete, manually verify:

1. **Photo tab — Direct mode**: Upload image → click "Detect Plates Directly" → see plate count → see 3-column comparison (Original, LapSRN, Real-ESRGAN) each with image + OCR text + confidence
2. **Photo tab — Cars mode**: Upload image → click "Detect Cars First" → select car from dropdown → click "Show Plate & OCR" → see car crop, plate crop, then 3-column comparison
3. **Video tab — Auto**: Upload video → click "Scan Video" → see car grid → select car → click "Auto Mode" → see 3-column comparison
4. **Video tab — Manual**: Upload video → click "Scan Video" → see car grid → select car → click "Manual Mode" → pick from candidates → see 3-column comparison
5. Both tabs end with the identical comparison layout
