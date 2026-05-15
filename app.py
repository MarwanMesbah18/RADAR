import streamlit as st

import config
from ui.photo_tab import render_photo_tab

st.set_page_config(
    page_title="RADAR - Vehicle Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

with st.sidebar:
    st.title("RADAR")
    st.caption("Real-Time Vehicle Analysis System")
    st.divider()

    st.subheader("Detection Settings")

    st.markdown("**Plate Detector** (YOLOv11m V1)")
    config.PLATE_CONFIDENCE = st.slider(
        "Plate Confidence",
        min_value=0.01, max_value=0.99, value=config.PLATE_CONFIDENCE, step=0.01,
        key="plate_conf",
    )

    st.markdown("**Car Detector** (YOLOv26s)")
    config.CAR_CONFIDENCE = st.slider(
        "Car Confidence",
        min_value=0.01, max_value=0.99, value=config.CAR_CONFIDENCE, step=0.01,
        key="car_conf",
    )

    st.markdown("**Seatbelt + Mobile** (YOLOv11m)")
    config.SEATBELT_CONFIDENCE = st.slider(
        "Seatbelt Confidence",
        min_value=0.01, max_value=0.99, value=config.SEATBELT_CONFIDENCE, step=0.01,
        key="seatbelt_conf",
    )

    st.markdown("**OCR Models** (V1, V2, V2 Weighted)")
    config.OCR_CONFIDENCE = st.slider(
        "OCR Confidence",
        min_value=0.01, max_value=0.99, value=config.OCR_CONFIDENCE, step=0.01,
        key="ocr_conf",
    )

    st.markdown("**Preprocessing**")
    config.PLATE_CROP_MARGIN = st.slider(
        "Plate Crop Margin (px)",
        min_value=0, max_value=100, value=config.PLATE_CROP_MARGIN, step=1,
        key="crop_margin",
    )

    st.divider()
    st.subheader("Model Status")
    st.markdown("**Detection**")
    for name, ready in [
        ("Plate Detector (YOLOv11m V1)", True),
        ("Car Detector (YOLOv26s)", True),
        ("Seatbelt + Mobile (YOLOv11m)", True),
    ]:
        st.markdown(f"{'🟢' if ready else '⚪'} {name}")

    st.markdown("**OCR Models**")
    for name, ready in [
        ("OCR V1 (YOLOv11m)", True),
        ("OCR V2 (YOLOv26m)", True),
        ("OCR V2 Weighted (YOLOv26m)", True),
    ]:
        st.markdown(f"{'🟢' if ready else '⚪'} {name}")

    st.markdown("**Enhancement**")
    for name, ready in [
        ("LapSRN (AI Light)", True),
        ("Real-ESRGAN (AI Heavy)", True),
    ]:
        st.markdown(f"{'🟢' if ready else '⚪'} {name}")

    st.divider()
    st.caption("v2.0 | RADAR Project")

render_photo_tab()
