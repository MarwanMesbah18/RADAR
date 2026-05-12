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
