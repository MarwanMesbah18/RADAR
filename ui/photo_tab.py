import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import glob

import config
from core.pipeline import (
    _load_image, detect_cars_step, analyze_plates_step,
    generate_interior_summary, run_multi_size_seatbelt,
    StepImages, VehicleAnalysis,
)
from core.plate_detector import detect_plates
from core.seatbelt_detector import detect_seatbelt, get_seatbelt_summary, draw_seatbelt_detections
from core.enhancement import enhance_lapsrn, enhance_realesrgan
from core.preprocessing import pil_to_cv2
from ui.display import (
    show_image, show_enhancement_comparison,
    show_interior_table,
    show_final_summary,
)

# Sample images folder
SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "photos")


def _reset():
    for key in list(st.session_state.keys()):
        if key in ("photo_upload",):
            continue
        if key.startswith("photo_") or key in ("selected_car", "tmp_file"):
            st.session_state.pop(key, None)


def _render_sample_images():
    """Show sample images as a compact clickable gallery."""
    if not os.path.isdir(SAMPLE_DIR):
        return

    samples = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.[jJ][pP][gG]"))
                     + glob.glob(os.path.join(SAMPLE_DIR, "*.[pP][nN][gG]"))
                     + glob.glob(os.path.join(SAMPLE_DIR, "*.[jJ][pP][eE][gG]")))

    if not samples:
        return

    with st.expander("Sample Images", expanded=False):
        # Compact grid: many columns so cards stick together
        cols_per_row = min(len(samples), 8)
        for i in range(0, len(samples), cols_per_row):
            row = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                with row[j]:
                    if idx < len(samples):
                        path = samples[idx]
                        st.image(path)
                        if st.button("📂", key=f"sample_{idx}",
                                     help=f"Load {os.path.basename(path)}"):
                            _reset()
                            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                            image = Image.open(path).convert("RGB")
                            image.save(tmp.name, format="PNG")
                            tmp.close()
                            st.session_state["tmp_file"] = tmp.name
                            st.session_state["photo_sample_selected"] = True
                            st.rerun()


def render_photo_tab():
    # ── Sample images ──
    _render_sample_images()

    # ── Upload ──
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"],
        key="photo_upload", on_change=_reset,
    )

    # Auto-analyze on upload
    if uploaded is not None and not st.session_state.get("photo_analyze_clicked"):
        if "tmp_file" not in st.session_state:
            image = Image.open(uploaded).convert("RGB")
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            image.save(tmp.name, format="PNG")
            tmp.close()
            st.session_state["tmp_file"] = tmp.name
            st.session_state["photo_analyze_clicked"] = True

    # Sample image selected
    if st.session_state.get("photo_sample_selected") and not st.session_state.get("photo_analyze_clicked"):
        st.session_state["photo_analyze_clicked"] = True

    if not st.session_state.get("photo_analyze_clicked") or "tmp_file" not in st.session_state:
        return

    tmp_path = st.session_state["tmp_file"]
    img = cv2.imread(tmp_path)
    if img is not None:
        show_image(img, "Original Image", width=3)

    st.markdown("---")

    # ── Step 1: Detect vehicles (run ONCE at cache-min confidence) ──
    if "photo_car_tracks_raw" not in st.session_state:
        pil_img, bgr_img = _load_image(tmp_path)
        with st.spinner("Detecting vehicles..."):
            car_tracks_raw, _ = detect_cars_step(bgr_img, conf=config.CAR_CONFIDENCE_CACHE)
        st.session_state["photo_pil"] = pil_img
        st.session_state["photo_bgr"] = bgr_img
        st.session_state["photo_car_tracks_raw"] = car_tracks_raw

    car_tracks_raw = st.session_state.get("photo_car_tracks_raw", [])

    # ── Filter by current CAR_CONFIDENCE slider ──
    car_tracks = [t for t in car_tracks_raw if t.confidence >= config.CAR_CONFIDENCE]

    # Invalidate selection if filtered count changed (selected car may have shifted)
    prev_count = st.session_state.get("photo_filtered_count")
    if prev_count is not None and len(car_tracks) != prev_count:
        st.session_state.pop("selected_car", None)
        # Clear per-car caches since the index mapping changed
        for k in ["car_vehicle", "car_seatbelt_summary",
                  "car_orig_dets", "car_lap_dets", "car_esrgan_dets",
                  "car_lap_raw", "car_esrgan_raw"]:
            st.session_state.pop(k, None)
        for k in list(st.session_state.keys()):
            if k.startswith("enh_comp_"):
                st.session_state.pop(k, None)
    st.session_state["photo_filtered_count"] = len(car_tracks)

    if not car_tracks:
        st.warning("No vehicles detected at current threshold.")
        _render_fallback_plates()
        return

    # Re-draw annotated image from filtered tracks
    bgr_img = st.session_state["photo_bgr"]
    annotated = bgr_img.copy()
    for i, car in enumerate(car_tracks):
        cx1, cy1, cx2, cy2 = [int(v) for v in car.bbox]
        cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), (0, 0, 255), 2)
        cv2.putText(annotated, f"{car.class_name} #{i + 1}",
                    (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    show_image(annotated, f"Detected {len(car_tracks)} vehicle(s)", width=5)

    # ── Car selection grid ──
    st.markdown("### Select a Vehicle")
    _render_car_grid(car_tracks, bgr_img)

    st.markdown("---")

    # ── Run analysis for selected car ──
    idx = st.session_state.get("selected_car")
    if idx is None or idx >= len(car_tracks):
        st.info("Click a vehicle above to run analysis on it.")
        return

    _render_car_analysis(car_tracks[idx], idx)


def _render_car_grid(car_tracks, bgr_img):
    """Show clickable car cards — just the car images."""
    cols_per_row = 4
    for i in range(0, len(car_tracks), cols_per_row):
        row = st.columns(cols_per_row)
        for j, car in enumerate(car_tracks[i:i + cols_per_row]):
            idx = i + j
            with row[j]:
                cx1, cy1, cx2, cy2 = [int(v) for v in car.bbox]
                h, w = bgr_img.shape[:2]
                crop = bgr_img[max(0, cy1):min(h, cy2), max(0, cx1):min(w, cx2)]
                st.markdown(f"**#{idx + 1}** {car.class_name}")
                if crop.size > 0:
                    st.image(crop, channels="BGR")
                if st.button("Select", key=f"car_btn_{idx}", use_container_width=True):
                    # Clear per-car analysis cache
                    for k in ["car_vehicle", "car_seatbelt_summary",
                              "car_orig_dets", "car_lap_dets", "car_esrgan_dets",
                              "car_lap_raw", "car_esrgan_raw"]:
                        st.session_state.pop(k, None)
                    # Clear OCR comparison cache
                    for k in list(st.session_state.keys()):
                        if k.startswith("enh_comp_"):
                            st.session_state.pop(k, None)
                    st.session_state["selected_car"] = idx


def _render_car_analysis(car, car_idx):
    """Run and display full analysis for one car — all stacked vertically."""
    st.markdown(f"## Vehicle #{car_idx + 1} — {car.class_name}")

    bgr_img = st.session_state["photo_bgr"]
    cx1, cy1, cx2, cy2 = [int(v) for v in car.bbox]
    h, w = bgr_img.shape[:2]
    car_crop = bgr_img[max(0, cy1):min(h, cy2), max(0, cx1):min(w, cx2)]

    if car_crop.size == 0:
        st.error("Could not crop vehicle region.")
        return

    # ── Interior Analysis (3 columns, each fills as it completes) ──
    st.markdown("### Interior Analysis")

    col1, col2, col3 = st.columns(3)

    # Column 1: Original — run ONCE at cache-min conf, filter on each rerun
    with col1:
        st.markdown("**Original**")
        if "car_orig_dets" not in st.session_state:
            with st.spinner("Detecting..."):
                orig_dets = detect_seatbelt(car_crop, conf=config.SEATBELT_CONFIDENCE_CACHE)
                st.session_state["car_orig_dets"] = orig_dets
        # Filter by current slider
        filtered = [d for d in st.session_state["car_orig_dets"]
                     if d.confidence >= config.SEATBELT_CONFIDENCE]
        if filtered:
            st.image(draw_seatbelt_detections(car_crop, filtered), channels="BGR")
        else:
            st.image(car_crop, channels="BGR")
        _render_person_columns(filtered)

    # Column 2: LapSRN — enhance once, detect once at cache-min, filter on each rerun
    with col2:
        st.markdown("**LapSRN (AI)**")
        if "car_lap_dets" not in st.session_state:
            with st.spinner("Enhancing & detecting..."):
                try:
                    lap_raw = enhance_lapsrn(car_crop)
                    if lap_raw is not None:
                        st.session_state["car_lap_raw"] = lap_raw
                        lap_dets = detect_seatbelt(lap_raw, conf=config.SEATBELT_CONFIDENCE_CACHE)
                    else:
                        lap_dets = []
                except Exception:
                    lap_dets = []
                st.session_state["car_lap_dets"] = lap_dets
        lap_raw = st.session_state.get("car_lap_raw")
        filtered = [d for d in st.session_state["car_lap_dets"]
                     if d.confidence >= config.SEATBELT_CONFIDENCE]
        if lap_raw is not None and filtered:
            st.image(draw_seatbelt_detections(lap_raw, filtered), channels="BGR")
        elif lap_raw is not None:
            st.image(lap_raw, channels="BGR")
        _render_person_columns(filtered)

    # Column 3: Real-ESRGAN — same pattern
    with col3:
        st.markdown("**Real-ESRGAN (AI)**")
        if "car_esrgan_dets" not in st.session_state:
            with st.spinner("Enhancing & detecting..."):
                try:
                    esrgan_raw = enhance_realesrgan(car_crop)
                    if esrgan_raw is not None:
                        st.session_state["car_esrgan_raw"] = esrgan_raw
                        esrgan_dets = detect_seatbelt(esrgan_raw, conf=config.SEATBELT_CONFIDENCE_CACHE)
                    else:
                        esrgan_dets = []
                except Exception:
                    esrgan_dets = []
                st.session_state["car_esrgan_dets"] = esrgan_dets
        esrgan_raw = st.session_state.get("car_esrgan_raw")
        filtered = [d for d in st.session_state["car_esrgan_dets"]
                     if d.confidence >= config.SEATBELT_CONFIDENCE]
        if esrgan_raw is not None and filtered:
            st.image(draw_seatbelt_detections(esrgan_raw, filtered), channels="BGR")
        elif esrgan_raw is not None:
            st.image(esrgan_raw, channels="BGR")
        _render_person_columns(filtered)

    # Merged summary
    all_dets = (st.session_state["car_orig_dets"]
                + st.session_state.get("car_lap_dets", [])
                + st.session_state.get("car_esrgan_dets", []))
    seatbelt_summary = get_seatbelt_summary(all_dets)
    st.session_state["car_seatbelt_summary"] = seatbelt_summary

    st.markdown("---")

    # ── Plate Detection & OCR ──
    st.markdown("### License Plate")

    if "car_vehicle" not in st.session_state:
        pil_img = st.session_state["photo_pil"]
        vehicle = VehicleAnalysis(
            car_index=car_idx,
            car_class=car.class_name,
            seatbelt_summary=seatbelt_summary,
            steps=StepImages(car_crop=car_crop.copy()),
        )
        with st.spinner("Detecting license plate and reading characters..."):
            h_img, w_img = bgr_img.shape[:2]
            analyze_plates_step(bgr_img, pil_img, [vehicle], w_img, h_img)
        st.session_state["car_vehicle"] = vehicle

    vehicle = st.session_state["car_vehicle"]

    if vehicle.plate_detection is None:
        st.info("No license plate detected on this vehicle.")
        return

    # Show plate crop
    st.markdown("**Plate Detected**")
    if vehicle.steps and vehicle.steps.plate_crop is not None:
        col_plate, col_conf = st.columns([2, 1])
        with col_plate:
            show_image(vehicle.steps.plate_crop, width=2)
        with col_conf:
            st.metric("Confidence", f"{vehicle.plate_detection.confidence:.0%}")

    st.markdown("---")

    # OCR comparison
    st.markdown("### OCR Comparison")
    plate_crop = vehicle.steps.plate_crop if vehicle.steps and vehicle.steps.plate_crop is not None else pil_to_cv2(vehicle.plate_detection.cropped_image)
    show_enhancement_comparison(plate_crop)


def _render_fallback_plates():
    """Direct plate detection when no cars found."""
    if "photo_direct_plates" not in st.session_state:
        with st.spinner("Trying direct plate detection..."):
            plates = detect_plates(st.session_state["photo_pil"], conf=config.PLATE_CONFIDENCE_CACHE)
            st.session_state["photo_direct_plates"] = plates

    plates = st.session_state.get("photo_direct_plates", [])
    # Filter by current slider
    filtered = [p for p in plates if p.confidence >= config.PLATE_CONFIDENCE]
    if filtered:
        for i, plate in enumerate(filtered):
            st.markdown(f"#### Plate #{i + 1} (confidence: {plate.confidence:.0%})")
            show_enhancement_comparison(pil_to_cv2(plate.cropped_image))


def _render_person_columns(detections):
    """Show person results in 2 columns: Passenger (left) | Driver (right)."""
    if detections is None:
        st.caption("Processing...")
        return
    if not detections:
        st.caption("No detections")
        return

    persons = [d for d in detections if d.class_id in (0, 1)]
    phone_det = next((d for d in detections if d.class_id == 4), None)
    has_mobile = phone_det is not None

    if not persons:
        if has_mobile:
            st.error(":white_check_mark: **Phone** detected")
        else:
            st.caption("No persons detected")
        return

    # Sort rightmost first (highest x = Driver)
    persons.sort(key=lambda d: d.bbox[0], reverse=True)

    if len(persons) == 1:
        _render_one_person(persons[0], "Driver", has_mobile, phone_det)
        return

    # 2+ persons: Passenger (left) | Driver (right)
    col_pass, col_drv = st.columns(2)

    driver = persons[0]
    passenger = persons[1]

    with col_pass:
        _render_one_person(passenger, "Passenger", has_phone=False, phone_det=None)

    with col_drv:
        _render_one_person(driver, "Driver", has_phone=has_mobile, phone_det=phone_det)


def _render_one_person(person, label, has_phone, phone_det=None):
    """Render one person's seatbelt + phone status with Safe/Not Safe."""
    has_belt = person.class_id == 1

    st.markdown(f"**{label}:**")

    # Seatbelt: detected means tick, not detected means nothing
    if has_belt:
        st.markdown(f":white_check_mark: Seatbelt `{person.confidence:.0%}`")
    else:
        st.markdown(f":x: Seatbelt")

    # Phone: detected means tick (with confidence), not detected means nothing
    if has_phone and phone_det is not None:
        st.markdown(f":white_check_mark: Phone `{phone_det.confidence:.0%}`")
    elif has_phone:
        st.markdown(f":white_check_mark: Phone")
    else:
        st.markdown(f":x: Phone")

    # Verdict — only "Safe" or "Not Safe — reason"
    is_safe = has_belt and not has_phone
    if is_safe:
        st.success("**Safe** :white_check_mark:")
    else:
        reasons = []
        if not has_belt:
            reasons.append("no seatbelt")
        if has_phone:
            reasons.append("phone in use")
        st.error(f"**Not Safe** :x: — {', '.join(reasons)}")
