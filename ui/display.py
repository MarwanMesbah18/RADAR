import streamlit as st
import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image as PILImage
import config
from core.enhancement import enhance_lapsrn, enhance_realesrgan
from core.plate_ocr import ocr_yolo
from core.plate_utils import separate_chars
import config


def show_image(img_bgr, caption=None, width=3, max_height=300):
    """Display a BGR image centered in constrained columns."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) if len(img_bgr.shape) == 3 else img_bgr
    if max_height > 0 and rgb.shape[0] > max_height:
        scale = max_height / rgb.shape[0]
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * scale), max_height))
    c1, c2, c3 = st.columns([1, width, 1])
    with c2:
        st.image(rgb, caption=caption)


# ── Seatbelt / Interior Display ──


def show_interior_table(orig_dets, orig_img, lap_dets, lap_img, esrgan_dets, esrgan_img):
    """Show 3-column table: Original | LapSRN | Real-ESRGAN with per-person results."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original**")
        if orig_img is not None:
            st.image(orig_img, channels="BGR")
        _render_person_results(orig_dets)

    with col2:
        st.markdown("**LapSRN (AI)**")
        if lap_img is not None:
            st.image(lap_img, channels="BGR")
        elif lap_dets is not None:
            st.caption("Enhancing...")
        _render_person_results(lap_dets)

    with col3:
        st.markdown("**Real-ESRGAN (AI)**")
        if esrgan_img is not None:
            st.image(esrgan_img, channels="BGR")
        elif esrgan_dets is not None:
            st.caption("Enhancing...")
        _render_person_results(esrgan_dets)


def _render_person_results(detections):
    """Show per-person status as checkmark/cross table with Safe/Not Safe verdict."""
    if detections is None:
        st.caption("Processing...")
        return
    if not detections:
        st.caption("No detections")
        return

    persons = [d for d in detections if d.class_id in (0, 1)]
    has_mobile = any(d.class_id == 4 for d in detections)
    has_seatbelt_obj = any(d.class_id == 2 for d in detections)

    if persons:
        # Sort rightmost first (highest x = Driver in Egyptian cars)
        persons.sort(key=lambda d: d.bbox[0], reverse=True)
        labels = _get_person_labels(len(persons))

        for i, person in enumerate(persons):
            label = labels[i]
            has_belt = person.class_id == 1 or (person.class_id == 0 and has_seatbelt_obj)

            # Determine if this person has a phone (only matters for driver)
            person_has_phone = False
            if i == 0:  # driver
                person_has_phone = has_mobile

            belt_icon = "✅" if has_belt else "❌"
            phone_icon = "❌" if person_has_phone else "✅"

            # Checkmark table
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"{belt_icon} Seatbelt")
            with c2:
                st.markdown(f"{phone_icon} No Phone")

            # Safety verdict
            is_safe = has_belt and not person_has_phone
            if is_safe:
                st.success(f"**{label}: Safe**")
            else:
                reasons = []
                if not has_belt:
                    reasons.append("No seatbelt")
                if person_has_phone:
                    reasons.append("Phone in use")
                st.error(f"**{label}: Not Safe** — {', '.join(reasons)}")

            if i < len(persons) - 1:
                st.markdown("")  # spacing between persons

    elif has_mobile:
        st.error("❌ **Phone** detected")

    if not persons and not has_mobile:
        st.caption("No persons detected")


def _get_person_labels(count):
    """Generate labels — rightmost in photo = Driver."""
    if count == 1:
        return ["Driver"]
    elif count == 2:
        return ["Driver", "Passenger"]
    else:
        labels = ["Driver", "Passenger"]
        for i in range(2, count):
            labels.append(f"Person #{i + 1}")
        return labels


def show_interior_text_summary(seatbelt_summary):
    """Show a friendly one-line text summary."""
    if not seatbelt_summary:
        return
    parts = []
    if seatbelt_summary.get("has_no_seatbelt"):
        parts.append("No seatbelt detected")
    elif seatbelt_summary.get("has_seatbelt"):
        parts.append("Seatbelt detected")
    if seatbelt_summary.get("has_mobile"):
        parts.append("Mobile phone in use")
    if parts:
        st.markdown(f"*{'. '.join(parts)}.*")
    else:
        st.markdown("*No violations detected.*")


# ── Plate OCR Comparison ──


# Per-class color palette (YOLO-style, for OCR character annotations)
_OCR_PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
    (49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (199, 146, 24), (255, 57, 0), (255, 156, 163), (148, 57, 255),
    (255, 97, 134), (120, 208, 255), (255, 208, 120), (0, 255, 56),
    (163, 255, 0), (255, 120, 208), (112, 31, 255), (178, 29, 255),
    (0, 56, 255), (0, 151, 255), (56, 255, 56), (208, 120, 255),
    (255, 178, 29), (255, 31, 112), (120, 255, 208), (255, 208, 255),
    (31, 255, 178), (178, 255, 31), (208, 255, 120), (255, 56, 208),
    (146, 23, 204), (219, 134, 61), (24, 199, 146), (163, 255, 156),
]


def _char_color(class_name):
    """Get a consistent BGR color for a character class."""
    return _OCR_PALETTE[hash(class_name) % len(_OCR_PALETTE)]


# Reverse mapping: Arabic → Franco (for annotation labels)
_ARABIC_TO_FRANCO = {v: k for k, v in config.FRANCO_TO_ARABIC.items()}


def _draw_ocr_annotations(base_img, detections):
    """Draw OCR annotations with per-class colors, using Franco class names."""
    ann = base_img.copy()
    if isinstance(ann, np.ndarray):
        ann = np.ascontiguousarray(ann)
    for d in detections:
        dx1, dy1, dx2, dy2 = [int(v) for v in d.bbox]
        color = _char_color(d.class_name)
        cv2.rectangle(ann, (dx1, dy1), (dx2, dy2), color, 2)
        # Use Franco name for label (same as YOLO's default plot)
        franco = _ARABIC_TO_FRANCO.get(d.class_name, d.class_name)
        label = f"{franco} {d.confidence:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(ann, (dx1, dy1 - th - 4), (dx1 + tw + 2, dy1), color, -1)
        cv2.putText(ann, label, (dx1 + 1, dy1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    # Convert BGR to RGB (YOLO plot returns RGB, so match that)
    if isinstance(ann, np.ndarray) and len(ann.shape) == 3 and ann.shape[2] == 3:
        ann = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
    return ann


def show_enhancement_comparison(plate_crop):
    """Show OCR comparison as a bordered table with live filtering.

    Enhancement images and OCR raw results are cached in session_state.
    On each rerun (slider change), only filtering and drawing happens — no model calls.
    """
    # ── Cache enhancement images (run ONCE) ──
    if "enh_comp_lapsrn" not in st.session_state:
        with st.spinner("Enhancing plate image..."):
            st.session_state["enh_comp_lapsrn"] = enhance_lapsrn(plate_crop)
            st.session_state["enh_comp_esrgan"] = enhance_realesrgan(plate_crop)

    lapsrn_img = st.session_state["enh_comp_lapsrn"]
    esrgan_img = st.session_state["enh_comp_esrgan"]

    models = [
        ("OCR V1", 1),
        ("OCR V2", 2),
        ("OCR V2 (Weighted)", 3),
    ]

    enhancements = [
        ("Original", plate_crop, "original"),
        ("LapSRN (AI)", lapsrn_img, "lapsrn"),
        ("Real-ESRGAN (AI)", esrgan_img, "esrgan"),
    ]

    # ── Cache raw OCR results (run ONCE at cache-min conf) ──
    if "enh_comp_ocr_raw" not in st.session_state:
        raw = {}
        for m_label, version in models:
            for e_label, e_img, e_key in enhancements:
                raw[(m_label, e_key)] = ocr_yolo(
                    e_img, model_version=version, conf=config.OCR_CONFIDENCE_CACHE
                )
        st.session_state["enh_comp_ocr_raw"] = raw

    # ── Filter by current OCR confidence slider ──
    conf = config.OCR_CONFIDENCE
    filtered_results = {}
    for (m_label, e_key), (dets, annotated_img) in st.session_state["enh_comp_ocr_raw"].items():
        filtered_dets = [d for d in dets if d.confidence >= conf]
        chars = separate_chars(filtered_dets)
        all_confs = [d.confidence for d in filtered_dets]
        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
        base_img = next(e_img for _, e_img, ek in enhancements if ek == e_key)
        # Use original YOLO annotated image if no filtering happened (all pass),
        # otherwise draw with per-class colors from filtered detections
        if len(filtered_dets) == len(dets) and annotated_img is not None:
            display_img = annotated_img
        else:
            display_img = _draw_ocr_annotations(base_img, filtered_dets)
        filtered_results[(m_label, e_key)] = {
            "text": chars.text or "—",
            "confidence": f"{avg_conf:.0%}",
            "display_image": display_img,
        }

    # CSS for bordered table
    st.markdown("""<style>
    .ocr-table {
        width: 100%;
        border-collapse: collapse;
    }
    .ocr-table th, .ocr-table td {
        border: 1px solid #555;
        text-align: center;
        padding: 10px;
        vertical-align: middle;
    }
    .ocr-table th {
        background-color: #262730;
        font-weight: bold;
        white-space: nowrap;
    }
    .ocr-table .row-header {
        width: 1%;
        white-space: nowrap;
    }
    .ocr-table td {
        height: 250px;
    }
    .ocr-table td img {
        max-height: 180px;
        width: auto;
    }
    </style>""", unsafe_allow_html=True)

    # Build HTML table
    html = '<table class="ocr-table">'
    # Header row
    html += '<tr><th></th>'
    for m_label, _ in models:
        html += f'<th>{m_label}</th>'
    html += '</tr>'

    # Data rows
    for e_label, e_img, e_key in enhancements:
        html += f'<tr><th class="row-header">{e_label}</th>'
        for m_label, _ in models:
            r = filtered_results[(m_label, e_key)]
            plate = r["text"]
            conf_str = r["confidence"]
            display_img = r["display_image"]

            # Convert to RGB for display
            if isinstance(display_img, np.ndarray):
                if len(display_img.shape) == 2:
                    rgb = cv2.cvtColor(display_img, cv2.COLOR_GRAY2RGB)
                elif display_img.shape[2] == 3:
                    # Check if already RGB (from YOLO plot) or BGR
                    rgb = display_img  # YOLO plot() returns RGB
                else:
                    rgb = display_img
            else:
                rgb = display_img

            buf = BytesIO()
            PILImage.fromarray(rgb).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            img_html = f'<img src="data:image/png;base64,{b64}" style="width:100%"/>'
            html += f'<td>{img_html}<br/><b>Plate:</b> <code>{plate}</code><br/><b>Confidence:</b> <code>{conf_str}</code></td>'
        html += '</tr>'

    html += '</table>'
    st.markdown(html, unsafe_allow_html=True)


# ── Final Summary ──


def show_final_summary(vehicle, plate_text=None, plate_confidence=None):
    """Show a clean summary card for one vehicle."""
    st.markdown("#### Vehicle Summary")

    # Vehicle info
    st.markdown(f"**Type:** {vehicle.car_class}")

    # Plate info
    if plate_text:
        st.markdown(f"**Plate Number:** `{plate_text}`")
        if plate_confidence:
            st.markdown(f"**Plate Confidence:** `{plate_confidence:.0%}`")
    else:
        st.markdown("**Plate:** No plate detected")

    # Interior violations
    summary = vehicle.seatbelt_summary
    if summary:
        violations = []
        if summary.get("has_no_seatbelt"):
            violations.append("No Seatbelt")
        if summary.get("has_mobile"):
            violations.append("Mobile Phone in Use")
        if violations:
            st.markdown("**Violations:** " + ", ".join(f"`{v}`" for v in violations))
        else:
            st.markdown("**Violations:** None detected")


# ── Kept for backward compatibility with video_tab ──


def show_seatbelt_badges(seatbelt_summary):
    """Show seatbelt/mobile violation badges (used by video tab)."""
    if not seatbelt_summary:
        return
    badges = []
    if seatbelt_summary.get("has_no_seatbelt"):
        badges.append(":red[**No Belt**]")
    if seatbelt_summary.get("has_seatbelt"):
        badges.append(":green[**Belt**]")
    if seatbelt_summary.get("has_mobile"):
        badges.append(":red[**Phone**]")
    if badges:
        st.markdown(" ".join(badges))


def show_multi_size_seatbelt(seatbelt_multi_result):
    """Show seatbelt detections at 3 enhancement levels (used by video tab)."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Original**")
        if seatbelt_multi_result.annotated_original is not None:
            st.image(seatbelt_multi_result.annotated_original, channels="BGR")
        else:
            st.caption("No detections")
    with col2:
        st.markdown("**LapSRN Enhanced**")
        if seatbelt_multi_result.annotated_lapsrn is not None:
            st.image(seatbelt_multi_result.annotated_lapsrn, channels="BGR")
        else:
            st.caption("No detections")
    with col3:
        st.markdown("**Real-ESRGAN Enhanced**")
        if seatbelt_multi_result.annotated_esrgan is not None:
            st.image(seatbelt_multi_result.annotated_esrgan, channels="BGR")
        else:
            st.caption("No detections")
