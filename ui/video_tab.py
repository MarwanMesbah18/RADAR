import streamlit as st
import tempfile

from pipelines.video_pipeline import scan_video_full
from pipelines.photo_pipeline import generate_interior_summary
from ui.display import (
    show_enhancement_comparison, show_seatbelt_badges, show_image,
    show_interior_text_summary, show_final_summary,
)


def _reset_scan():
    for key in ["video_cars", "video_selected_car", "video_selected_plate"]:
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

    # ── Scan Video ──
    scan_clicked = st.button("Scan Video", type="primary", use_container_width=True)

    if scan_clicked:
        _reset_scan()
        with st.status("Scanning video...", expanded=True) as scan_status:
            progress_bar = st.progress(0, text="Scanning video for vehicles...")

            def on_progress(fraction, info):
                progress_bar.progress(
                    fraction,
                    text=f"Frame {info['frame']}/{info['total']} — {info['cars']} vehicles, {info['plates']} plates",
                )

            cars = scan_video_full(tfile.name, progress_callback=on_progress)
            st.session_state["video_cars"] = cars

            progress_bar.progress(1.0, text=f"Done! Found {len(cars)} unique vehicles.")
            scan_status.update(
                label=f"Scan complete: {len(cars)} vehicles, "
                      f"{sum(1 for c in cars if c.has_plate)} with plates",
                state="complete", expanded=False,
            )

    cars = st.session_state.get("video_cars")
    if not cars:
        if not scan_clicked:
            st.info("Upload a video and click **Scan Video** to start.")
        return

    # ── All Vehicles ──
    st.markdown("### All Vehicles")
    st.markdown(f"Found **{len(cars)}** unique vehicles.")
    _render_car_grid(cars, key_prefix="all")

    # ── Cars with Plates ──
    cars_with_plates = [c for c in cars if c.has_plate]
    if cars_with_plates:
        st.markdown("---")
        st.markdown("### Cars with Plates")
        st.markdown(f"**{len(cars_with_plates)}** vehicles with detected plates.")
        _render_car_grid(cars_with_plates, key_prefix="plates")

    # ── Selected Car Analysis ──
    st.markdown("---")
    selected_idx = st.session_state.get("video_selected_car")
    if selected_idx is not None:
        car = cars[selected_idx]
        _render_car_analysis(car)
    else:
        st.info("Click a vehicle above to see its analysis.")


def _render_car_grid(cars, key_prefix="all"):
    """Render clickable car cards in a grid."""
    cols_per_row = 4
    for i in range(0, len(cars), cols_per_row):
        row = st.columns(cols_per_row)
        for j, car in enumerate(cars[i:i + cols_per_row]):
            idx = i + j
            with row[j]:
                plate_badge = "🟢 Plate" if car.has_plate else "🔴 No plate"
                st.markdown(f"**#{idx + 1}** {car.car_class} — {plate_badge}")
                show_seatbelt_badges(car.seatbelt_summary)
                if car.best_crop is not None:
                    st.image(car.best_crop, channels="BGR")
                if st.button("Select", key=f"{key_prefix}_sel_{idx}", use_container_width=True):
                    st.session_state["video_selected_car"] = idx


def _render_car_analysis(car):
    """Show full analysis for a selected car: interior + plate candidates."""
    st.markdown(f"### Vehicle Analysis — {car.car_class}")

    # Interior detection
    col_crop, col_info = st.columns([1, 2])
    with col_crop:
        st.markdown("**Vehicle**")
        if car.best_crop is not None:
            st.image(car.best_crop, channels="BGR")

    with col_info:
        st.markdown("**Interior Detection**")
        show_seatbelt_badges(car.seatbelt_summary)
        summary_text = generate_interior_summary(car.seatbelt_summary)
        show_interior_text_summary(summary_text)

    st.markdown("---")

    # Plate candidates
    if not car.has_plate or not car.plate_candidates:
        st.info("No plate detected for this vehicle.")
        return

    st.markdown(f"### Plate Candidates ({len(car.plate_candidates)})")
    st.markdown("Click a plate to see the OCR comparison.")

    cols_per_row = 3
    for i in range(0, len(car.plate_candidates), cols_per_row):
        row = st.columns(cols_per_row)
        for j, c in enumerate(car.plate_candidates[i:i + cols_per_row]):
            idx = i + j
            with row[j]:
                st.markdown(f"**Frame {c.frame_num}**")
                st.markdown(f"Area: `{c.crop_area:.0f} px²`")
                st.image(c.plate_crop, channels="BGR")
                if st.button("Select Plate", key=f"plate_sel_{idx}", use_container_width=True):
                    st.session_state["video_selected_plate"] = idx

    st.markdown("---")

    # Selected plate OCR
    plate_idx = st.session_state.get("video_selected_plate")
    if plate_idx is not None and plate_idx < len(car.plate_candidates):
        chosen = car.plate_candidates[plate_idx]

        col_car, col_plate = st.columns(2)
        with col_car:
            st.markdown("**Vehicle at Frame**")
            st.image(chosen.car_crop, channels="BGR")
        with col_plate:
            st.markdown("**Plate**")
            st.image(chosen.plate_crop, channels="BGR")

        st.markdown("---")
        st.markdown("### Enhancement & OCR Comparison")
        show_enhancement_comparison(chosen.plate_crop)
    else:
        st.info("Click a plate candidate above to see the OCR comparison.")
