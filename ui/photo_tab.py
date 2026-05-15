import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile

from pipelines.photo_pipeline import (
    _load_image, detect_cars_step, analyze_plates_step,
    generate_interior_summary, run_multi_size_seatbelt,
    StepImages, VehicleAnalysis,
)
from core.plate_detector import detect_plates
from core.seatbelt_detector import detect_seatbelt, get_seatbelt_summary, draw_seatbelt_detections
from core.enhancement import enhance_lapsrn, enhance_realesrgan
from utils.preprocessing import pil_to_cv2
from ui.display import (
    show_image, show_enhancement_comparison,
    show_interior_table,
    show_final_summary,
)


def _reset():
    for key in list(st.session_state.keys()):
        if key in ("photo_upload",):
            continue
        if key.startswith("photo_") or key in ("selected_car", "tmp_file"):
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
        show_image(img, "Original Image", width=3)

    st.markdown("---")

    if st.button("Analyze", type="primary", use_container_width=True):
        _reset()
        st.session_state["photo_analyze_clicked"] = True

    if not st.session_state.get("photo_analyze_clicked"):
        return

    # ── Step 1: Detect vehicles ──
    if "photo_cars_detected" not in st.session_state:
        pil_img, bgr_img = _load_image(tmp_path)
        with st.spinner("Detecting vehicles..."):
            car_tracks, annotated = detect_cars_step(bgr_img)
        st.session_state["photo_pil"] = pil_img
        st.session_state["photo_bgr"] = bgr_img
        st.session_state["photo_car_tracks"] = car_tracks
        st.session_state["photo_annotated"] = annotated
        st.session_state["photo_cars_detected"] = True

    car_tracks = st.session_state.get("photo_car_tracks", [])

    if not car_tracks:
        st.warning("No vehicles detected.")
        _render_fallback_plates()
        return

    # Show detected vehicles
    show_image(st.session_state["photo_annotated"],
               f"Detected {len(car_tracks)} vehicle(s)", width=5)

    # ── Car selection grid ──
    st.markdown("### Select a Vehicle")
    bgr_img = st.session_state["photo_bgr"]
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
                    for k in ["car_vehicle", "car_seatbelt_summary",
                              "car_orig_dets", "car_orig_img",
                              "car_lap_dets", "car_lap_img",
                              "car_esrgan_dets", "car_esrgan_img"]:
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

    # Column 1: Original (runs fast, fills first)
    with col1:
        st.markdown("**Original**")
        if "car_orig_dets" not in st.session_state:
            with st.spinner("Detecting..."):
                orig_dets = detect_seatbelt(car_crop)
                orig_img = draw_seatbelt_detections(car_crop, orig_dets) if orig_dets else None
                st.session_state["car_orig_dets"] = orig_dets
                st.session_state["car_orig_img"] = orig_img
        if st.session_state["car_orig_img"] is not None:
            st.image(st.session_state["car_orig_img"], channels="BGR")
        _render_person_columns(st.session_state["car_orig_dets"])

    # Column 2: LapSRN (runs after original)
    with col2:
        st.markdown("**LapSRN (AI)**")
        if "car_lap_dets" not in st.session_state:
            with st.spinner("Enhancing with AI..."):
                try:
                    lap_enhanced = enhance_lapsrn(car_crop)
                    if lap_enhanced is not None:
                        lap_dets = detect_seatbelt(lap_enhanced)
                        lap_img = draw_seatbelt_detections(lap_enhanced, lap_dets) if lap_dets else None
                    else:
                        lap_dets = []
                        lap_img = None
                except Exception:
                    lap_dets = []
                    lap_img = None
                st.session_state["car_lap_dets"] = lap_dets
                st.session_state["car_lap_img"] = lap_img
        if st.session_state["car_lap_img"] is not None:
            st.image(st.session_state["car_lap_img"], channels="BGR")
        _render_person_columns(st.session_state["car_lap_dets"])

    # Column 3: Real-ESRGAN (runs after LapSRN)
    with col3:
        st.markdown("**Real-ESRGAN (AI)**")
        if "car_esrgan_dets" not in st.session_state:
            with st.spinner("Enhancing with AI..."):
                try:
                    esrgan_enhanced = enhance_realesrgan(car_crop)
                    if esrgan_enhanced is not None:
                        esrgan_dets = detect_seatbelt(esrgan_enhanced)
                        esrgan_img = draw_seatbelt_detections(esrgan_enhanced, esrgan_dets) if esrgan_dets else None
                    else:
                        esrgan_dets = []
                        esrgan_img = None
                except Exception:
                    esrgan_dets = []
                    esrgan_img = None
                st.session_state["car_esrgan_dets"] = esrgan_dets
                st.session_state["car_esrgan_img"] = esrgan_img
        if st.session_state["car_esrgan_img"] is not None:
            st.image(st.session_state["car_esrgan_img"], channels="BGR")
        _render_person_columns(st.session_state["car_esrgan_dets"])

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
            plates = detect_plates(st.session_state["photo_pil"])
            st.session_state["photo_direct_plates"] = plates

    plates = st.session_state.get("photo_direct_plates", [])
    if plates:
        for i, plate in enumerate(plates):
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
