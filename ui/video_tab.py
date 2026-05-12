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