import streamlit as st
import tempfile

from pipelines.video_pipeline import scan_video_cars, find_plate_crops
from ui.display import show_enhancement_comparison, show_seatbelt_badges


def _reset_scan():
    for key in ["video_cars", "video_selected_car", "video_plate_mode",
                "video_plate_crop", "video_candidates", "video_selected_plate"]:
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

    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        with st.expander("Original Video", expanded=False):
            st.video(tfile.name)

    st.markdown("---")

    # ── Step 1: Scan Video ──
    if not st.button("🎬 Scan Video", type="primary", use_container_width=True):
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
    """Show car grid as clickable cards and let user pick one."""
    cars = st.session_state["video_cars"]

    st.markdown("### Detected Vehicles")
    st.markdown(f"Found **{len(cars)}** unique vehicles.")

    # Car cards: 4 per row
    cols_per_row = 4
    for i in range(0, len(cars), cols_per_row):
        row = st.columns(cols_per_row)
        for j, car in enumerate(cars[i:i + cols_per_row]):
            idx = i + j
            with row[j]:
                st.markdown(f"**Car #{idx + 1}** ({car.car_class})")
                show_seatbelt_badges(car.seatbelt_summary)
                if car.best_crop is not None:
                    st.image(car.best_crop, channels="BGR")
                if st.button(f"Select", key=f"car_sel_{idx}", use_container_width=True):
                    st.session_state["video_selected_car"] = idx

    st.markdown("---")

    selected_idx = st.session_state.get("video_selected_car")
    if selected_idx is None:
        st.info("Click a car above to analyze its plate.")
        return

    car = cars[selected_idx]

    # Mode buttons
    col_auto, col_manual = st.columns(2)
    with col_auto:
        st.button("⚡ Auto Mode (Coming Soon)", disabled=True, use_container_width=True)
    with col_manual:
        if st.button("🎯 Manual Mode", type="primary", use_container_width=True):
            st.session_state["video_plate_mode"] = "manual"

    mode = st.session_state.get("video_plate_mode")
    if mode == "manual":
        _render_manual_mode(video_path, car)


def _render_manual_mode(video_path, car):
    """Manual: show plate candidate images as clickable cards."""
    st.markdown("### Manual Mode — Select best plate")

    if "video_candidates" not in st.session_state:
        with st.spinner("Scanning frames for plate candidates..."):
            candidates = find_plate_crops(video_path, car, top_n=5)
            st.session_state["video_candidates"] = candidates

    candidates = st.session_state["video_candidates"]

    if not candidates:
        st.warning(f"No plate detected for this car.")
        return

    # Plate candidate cards: 3 per row
    cols_per_row = 3
    for i in range(0, len(candidates), cols_per_row):
        row = st.columns(cols_per_row)
        for j, c in enumerate(candidates[i:i + cols_per_row]):
            idx = i + j
            with row[j]:
                st.markdown(f"**Frame {c.frame_num}**  ")
                st.markdown(f"Area: `{c.crop_area:.0f} px²`")
                st.image(c.plate_crop, channels="BGR")
                if st.button("Select", key=f"plate_sel_{idx}", use_container_width=True):
                    st.session_state["video_selected_plate"] = idx

    st.markdown("---")

    plate_idx = st.session_state.get("video_selected_plate")
    if plate_idx is None:
        st.info("Click a plate candidate above to see the OCR comparison.")
        return

    chosen = candidates[plate_idx]

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
