import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
import os
import time

import config
from pipelines.photo_pipeline import analyze_photo
from pipelines.video_pipeline import process_video

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RADAR - Vehicle Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar: Settings ────────────────────────────────────────
with st.sidebar:
    st.title("RADAR")
    st.caption("Real-Time Vehicle Analysis System")
    st.divider()

    st.subheader("Detection Settings")
    plate_conf = st.slider(
        "Plate Detection Confidence",
        min_value=0.05, max_value=0.9, value=config.PLATE_CONFIDENCE, step=0.05,
    )
    ocr_conf = st.slider(
        "OCR Confidence",
        min_value=0.05, max_value=0.9, value=config.OCR_CONFIDENCE, step=0.05,
    )
    crop_margin = st.slider(
        "Plate Crop Margin (px)",
        min_value=0, max_value=50, value=config.PLATE_CROP_MARGIN, step=2,
    )

    config.PLATE_CONFIDENCE = plate_conf
    config.OCR_CONFIDENCE = ocr_conf
    config.PLATE_CROP_MARGIN = crop_margin

    st.divider()
    st.subheader("Model Status")
    for name, ready in [
        ("Plate Detector (YOLOv11m)", True),
        ("Plate OCR (YOLOv11m)", True),
        ("Car Detector + Tracker (YOLOv11n)", True),
        ("Seatbelt (Coming Soon)", False),
    ]:
        icon = "🟢" if ready else "⚪"
        st.markdown(f"{icon} {name}")

    st.divider()
    st.caption(f"v1.0 | RADAR Project")

# ── Main tabs ────────────────────────────────────────────────
tab_photo, tab_video = st.tabs(["📷 Photo Mode", "🎬 Video Mode"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PHOTO TAB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_photo:
    uploaded = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "bmp", "tiff"],
        key="photo_upload",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")

        # Save to temp as PNG (lossless) to preserve quality
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(tmp.name, format="PNG")
        tmp.close()

        # Step 1: Show uploaded image
        st.subheader("Step 1 — Uploaded Image")
        st.image(image, use_container_width=True)

        # Analyze button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing..."):
                result = analyze_photo(tmp.name)

            if result.plates_found == 0:
                st.warning("No license plates detected in this image.")
            else:
                v = result.vehicles[0]
                steps = result.steps

                # Step 2: Plate detected on original
                st.subheader("Step 2 — Plate Detected")
                if steps.plate_detected is not None:
                    st.image(
                        cv2.cvtColor(steps.plate_detected, cv2.COLOR_BGR2RGB),
                        caption=f"Plate bounding box (confidence: {v.plate_detection.confidence:.0%})",
                        use_container_width=True,
                    )

                # Step 3: Cropped plate
                st.subheader("Step 3 — Cropped Plate")
                if steps.plate_crop is not None:
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c2:
                        st.image(
                            cv2.cvtColor(steps.plate_crop, cv2.COLOR_BGR2RGB),
                            caption="Cropped license plate region",
                        )

                # Step 4: YOLO OCR
                st.subheader("Step 4 — YOLO OCR Results")
                if steps.yolo_ocr is not None:
                    c1, c2, c3 = st.columns([1, 2, 1])
                    with c2:
                        st.image(
                            cv2.cvtColor(steps.yolo_ocr, cv2.COLOR_BGR2RGB),
                            caption=f"{len(v.yolo_detections)} characters detected",
                        )

                # Step 5: Vehicle details
                st.subheader("Step 5 — Vehicle Details")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### OCR Results")
                    st.markdown(f"**Numbers:** `{' '.join(v.yolo_numbers) if v.yolo_numbers else '—'}`")
                    st.markdown(f"**Letters:** `{' '.join(v.yolo_letters) if v.yolo_letters else '—'}`")
                    if v.plate_ocr:
                        st.markdown(f"**Full Text:** `{v.plate_ocr.text}`")
                        st.markdown(f"**OCR Confidence:** `{v.plate_ocr.confidence:.0%}`")

                with col2:
                    st.markdown("#### Detection Info")
                    st.markdown(f"**Plate Confidence:** `{v.plate_detection.confidence:.0%}`")
                    st.markdown(f"**Characters Found:** `{len(v.yolo_detections)}`")

                st.divider()

                # Summary
                col_a, col_b, col_c = st.columns(3)
                plate_display = v.plate_ocr.text if v.plate_ocr else "—"
                conf_display = f"{v.plate_ocr.confidence:.0%}" if v.plate_ocr else "—"

                col_a.metric("License Plate", plate_display)
                col_b.metric("OCR Confidence", conf_display)
                col_c.metric("Processing Time", f"{result.processing_time:.2f}s")

            # Cleanup temp file
            os.unlink(tmp.name)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  VIDEO TAB
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_video:
    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"],
        key="video_upload",
    )

    if uploaded_video is not None:
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        tfile.close()

        # Show original video
        st.subheader("Original Video")
        st.video(tfile.name)

        if st.button("🎬 Process Video", type="primary", use_container_width=True):
            progress_bar = st.progress(0, text="Processing...")

            def on_progress(fraction, info):
                progress_bar.progress(
                    fraction,
                    text=f"Frame {info['frame']}/{info['total']} — {info['plates']} plates found",
                )

            with st.spinner("Processing video..."):
                output_path, stats, unique_plates = process_video(
                    tfile.name,
                    progress_callback=on_progress,
                )

            progress_bar.progress(1.0, text="Done!")

            # Show results
            st.subheader("Processed Video")
            st.video(output_path)

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Unique Plates", stats.unique_plates)
            col2.metric("Total Detections", stats.plates_detected)
            col3.metric("Frames Processed", stats.processed_frames)
            col4.metric("Processing FPS", f"{stats.processing_fps:.1f}")
            col5.metric("Total Time", f"{stats.elapsed_time:.1f}s")

            # Plate results table
            if unique_plates:
                st.subheader("Detected Plates")
                for i, plate in enumerate(unique_plates, 1):
                    with st.expander(
                        f"#{i} — {plate.plate_text}  |  "
                        f"Confidence: {plate.best_confidence:.0%}  |  "
                        f"Car #{plate.car_track_id} ({plate.car_class})  |  "
                        f"Frame {plate.best_frame_num}  |  "
                        f"Seen {plate.total_detections}x"
                    ):
                        if plate.best_frame_image is not None:
                            st.image(
                                cv2.cvtColor(plate.best_frame_image, cv2.COLOR_BGR2RGB),
                                caption=f"Best capture — Frame {plate.best_frame_num}",
                                use_container_width=True,
                            )
                        else:
                            st.info("No frame snapshot available.")

            # Save button
            with open(output_path, "rb") as f:
                st.download_button(
                    label="💾 Download Processed Video",
                    data=f.read(),
                    file_name="radar_output.mp4",
                    mime="video/mp4",
                    use_container_width=True,
                )
