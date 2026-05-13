import streamlit as st
import cv2
import numpy as np
from core.enhancement import enhance_lapsrn, enhance_realesrgan
from core.plate_ocr import ocr_plate


def show_image(img_bgr, caption=None, width=3, max_height=300):
    """Display a BGR image centered in constrained columns.

    width: 2=narrow (crops), 3=medium, 5=wide (full frames)
    max_height: max pixel height before downscaling (0 = no limit)
    """
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if len(img_bgr.shape) == 3 else img_bgr
    if max_height > 0 and rgb.shape[0] > max_height:
        scale = max_height / rgb.shape[0]
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * scale), max_height))
    c1, c2, c3 = st.columns([1, width, 1])
    with c2:
        st.image(rgb, caption=caption)


def show_vehicle_analysis(v):
    """Display the full analysis pipeline for a single vehicle."""
    steps = v.steps

    if v.plate_detection is None:
        st.info(f"No plate detected on this {v.car_class}.")
        if steps and steps.car_crop is not None:
            show_image(steps.car_crop, f"Cropped {v.car_class} — no plate found", width=2)
        return

    if steps.plate_detected is not None:
        st.markdown("**Vehicle + Plate Detection**")
        show_image(steps.plate_detected,
                   f"Plate bbox (confidence: {v.plate_detection.confidence:.0%})", width=5)

    if steps.car_crop is not None:
        st.markdown("**Vehicle Crop**")
        show_image(steps.car_crop, f"Cropped {v.car_class}", width=2)

    st.markdown("**Plate Crop**")
    if steps.plate_crop is not None:
        show_image(steps.plate_crop, "Cropped license plate", width=2)

    st.markdown("**OCR Results**")
    if steps.yolo_ocr is not None:
        show_image(steps.yolo_ocr, f"{len(v.yolo_detections)} characters detected", width=2)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Numbers:** `{' '.join(v.yolo_numbers) if v.yolo_numbers else '—'}`")
        st.markdown(f"**Letters:** `{' '.join(v.yolo_letters) if v.yolo_letters else '—'}`")
    with col2:
        st.markdown(f"**Full Plate:** `{v.plate_ocr.text if v.plate_ocr else '—'}`")
        if v.plate_ocr:
            st.markdown(f"**Confidence:** `{v.plate_ocr.confidence:.0%}`")


def _render_model_row(plate_crop, lapsrn_img, esrgan_img, label, version):
    """Render one 3-column row (Original + LapSRN + Real-ESRGAN) for a given OCR model version."""
    col_orig, col_lap, col_esrgan = st.columns(3)

    ocr_orig = ocr_plate(plate_crop, model_version=version)
    ocr_lap = ocr_plate(lapsrn_img, model_version=version)
    ocr_esrgan = ocr_plate(esrgan_img, model_version=version)

    with col_orig:
        st.markdown("**Original**")
        if ocr_orig.annotated_image is not None:
            st.image(ocr_orig.annotated_image)
        else:
            st.image(plate_crop, channels="BGR")
        st.markdown(f"Plate: `{ocr_orig.text or '—'}`")
        st.markdown(f"Confidence: `{ocr_orig.confidence:.0%}`")

    with col_lap:
        st.markdown("**LapSRN (AI)**")
        if ocr_lap.annotated_image is not None:
            st.image(ocr_lap.annotated_image)
        else:
            st.image(lapsrn_img, channels="BGR")
        st.markdown(f"Plate: `{ocr_lap.text or '—'}`")
        st.markdown(f"Confidence: `{ocr_lap.confidence:.0%}`")

    with col_esrgan:
        st.markdown("**Real-ESRGAN (AI)**")
        if ocr_esrgan.annotated_image is not None:
            st.image(ocr_esrgan.annotated_image)
        else:
            st.image(esrgan_img, channels="BGR")
        st.markdown(f"Plate: `{ocr_esrgan.text or '—'}`")
        st.markdown(f"Confidence: `{ocr_esrgan.confidence:.0%}`")


def show_enhancement_comparison(plate_crop):
    """Show 3-column enhancement + OCR comparison across all OCR models.

    Each model version gets its own row. Columns: Original, LapSRN, Real-ESRGAN.
    Only annotated images are shown (plain image as fallback).
    """
    lapsrn_img = enhance_lapsrn(plate_crop)
    esrgan_img = enhance_realesrgan(plate_crop)

    models = [
        ("OCR Model V1", 1),
        ("OCR Model V2", 2),
        ("OCR Model V2 (Weighted-3)", 3),
    ]

    for i, (label, version) in enumerate(models):
        if i > 0:
            st.markdown("---")
        st.markdown(f"##### {label}")
        _render_model_row(plate_crop, lapsrn_img, esrgan_img, label, version)
