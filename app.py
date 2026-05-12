import streamlit as st

import config
from ui.photo_tab import render_photo_tab
from ui.video_tab import render_video_tab

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
    config.PLATE_CONFIDENCE = st.slider(
        "Plate Detection Confidence",
        min_value=0.05, max_value=0.9, value=config.PLATE_CONFIDENCE, step=0.05,
    )
    config.OCR_CONFIDENCE = st.slider(
        "OCR Confidence",
        min_value=0.05, max_value=0.9, value=config.OCR_CONFIDENCE, step=0.05,
    )
    config.PLATE_CROP_MARGIN = st.slider(
        "Plate Crop Margin (px)",
        min_value=0, max_value=50, value=config.PLATE_CROP_MARGIN, step=2,
    )

    st.divider()
    st.subheader("Model Status")
    for name, ready in [
        ("Car Detector (YOLOv11n COCO)", True),
        ("Car Tracker (ByteTrack)", True),
        ("Plate Detector (YOLOv11m)", True),
        ("Plate OCR (YOLOv11m)", True),
        ("Seatbelt (Coming Soon)", False),
    ]:
        st.markdown(f"{'🟢' if ready else '⚪'} {name}")

    st.divider()
    st.caption("v1.0 | RADAR Project")

tab_photo, tab_video = st.tabs(["📷 Photo Mode", "🎬 Video Mode"])

with tab_photo:
    render_photo_tab()

with tab_video:
    render_video_tab()
