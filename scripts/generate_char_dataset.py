"""
Final YOLO26m Character Dataset Builder for RADAR OCR training.

Combines:
  1. Existing "egyptian car plates.yolo26" dataset (3878 real plate images)
  2. Synthetic composites from EALPR characters (with 4 quality variations)

Outputs a properly split (70/20/10) YOLO-format dataset ready for training.

Usage:
  python scripts/generate_char_dataset.py --samples 5     # preview only
  python scripts/generate_char_dataset.py --count 1500    # full build
"""

import argparse
import os
import random
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARS_DIR = os.path.join(
    BASE_DIR, "Datasets", "EALPR-master",
    "EALPR- LP characters dataset", "Characters"
)
EXISTING_DATASET = os.path.join(
    BASE_DIR, "Datasets", "egyptian car plates.yolo26"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "Datasets", "characters_final")

# ---------------------------------------------------------------------------
# Class mapping — MUST match the existing dataset's data.yaml order exactly
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '7aa', '8', '9',
    'Taa', 'Thaa', 'ain', 'alif', 'baa', 'daad', 'daal', 'faa',
    'ghayn', 'haa', 'jeem', 'kaaf', 'khaa', 'laam', 'meem', 'noon',
    'qaaf', 'raa', 'saad', 'seen', 'sheen', 'taa', 'thaa', 'waw',
    'yaa', 'zaal', 'zay'
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"

from config import FRANCO_TO_ARABIC

ARABIC_TO_FRANCO = {v: k for k, v in FRANCO_TO_ARABIC.items()}
ARABIC_TO_FRANCO["أ"] = "alif"
ARABIC_TO_FRANCO["ى"] = "yaa"

DIGIT_FRANCO = [str(i) for i in range(10)]
LETTER_FRANCO = [n for n in CLASS_NAMES if n not in DIGIT_FRANCO]

# ---------------------------------------------------------------------------
# Character image indexing
# ---------------------------------------------------------------------------

def index_characters(chars_dir: str) -> dict:
    """Scan EALPR Characters folder, group image paths by Franco name."""
    char_to_paths = {}
    for fname in sorted(os.listdir(chars_dir)):
        if not fname.endswith(".png"):
            continue
        parts = fname.rsplit("-", 2)
        if len(parts) < 3:
            continue
        arabic_char = parts[-2]
        if arabic_char in ARABIC_DIGITS:
            franco = str(ARABIC_DIGITS.index(arabic_char))
        else:
            franco = ARABIC_TO_FRANCO.get(arabic_char)
            if franco is None:
                continue
        path = os.path.join(chars_dir, fname)
        char_to_paths.setdefault(franco, []).append(path)
    return char_to_paths


# ---------------------------------------------------------------------------
# Image compositing
# ---------------------------------------------------------------------------

def paste_char_alpha(canvas, char_img, x, y):
    """Paste RGBA character image onto BGR canvas at (x, y)."""
    h, w = char_img.shape[:2]
    if char_img.shape[2] == 4:
        bgr = char_img[:, :, :3]
        alpha = char_img[:, :, 3] / 255.0
    else:
        bgr = char_img
        alpha = np.ones((h, w), dtype=np.float32)

    ch, cw = canvas.shape[:2]
    x2, y2 = min(x + w, cw), min(y + h, ch)
    sx, sy = max(0, -x), max(0, -y)
    if x2 <= x or y2 <= y or sx >= w or sy >= h:
        return (x, y, 0, 0)

    roi = canvas[max(y, 0):y2, max(x, 0):x2]
    a = alpha[sy:sy + roi.shape[0], sx:sx + roi.shape[1], np.newaxis]
    blended = (bgr[sy:sy + roi.shape[0], sx:sx + roi.shape[1]] * a +
               roi * (1 - a)).astype(np.uint8)
    canvas[max(y, 0):y2, max(x, 0):x2] = blended

    return (max(x, 0), max(y, 0), x2 - max(x, 0), y2 - max(y, 0))


def _scale_char_to_fit(char_img, slot_w, slot_h):
    """Scale character to fit within slot, maintaining aspect ratio."""
    h, w = char_img.shape[:2]
    if w == 0 or h == 0:
        return char_img
    scale = min(slot_w / w, slot_h / h) * 0.85
    return cv2.resize(char_img, (max(1, int(w * scale)), max(1, int(h * scale))),
                      interpolation=cv2.INTER_CUBIC)


def generate_composite(char_paths, rng):
    """Create one synthetic image: 3 numbers (left) + 3 letters (right)."""
    canvas_w, canvas_h = 400, 160
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    annotations = []

    digits = rng.choices(DIGIT_FRANCO, k=3)
    letters = rng.choices(LETTER_FRANCO, k=3)

    margin_x, margin_y, col_gap = 15, 15, 25
    col_w = (canvas_w - 2 * margin_x - col_gap) // 2
    slot_w = col_w // 3 - 4
    slot_h = canvas_h - 2 * margin_y

    def place_in_column(column_x, chars):
        for i, franco in enumerate(chars):
            paths = char_paths.get(franco, [])
            if not paths:
                continue
            char_img = cv2.imread(rng.choice(paths), cv2.IMREAD_UNCHANGED)
            if char_img is None:
                continue
            char_img = _scale_char_to_fit(char_img, slot_w, slot_h)
            h, w = char_img.shape[:2]
            slot_x = column_x + i * (col_w // 3)
            x = slot_x + (slot_w - w) // 2 + 2
            y = margin_y + (slot_h - h) // 2
            bbox = paste_char_alpha(canvas, char_img, x, y)
            if bbox[2] > 0 and bbox[3] > 0:
                annotations.append((franco, *bbox))

    place_in_column(margin_x, digits)
    place_in_column(margin_x + col_w + col_gap, letters)
    return canvas, annotations


# ---------------------------------------------------------------------------
# Quality variations
# ---------------------------------------------------------------------------

def apply_augmentation(image, rng):
    """Random degradation: blur, noise, JPEG artifacts, brightness, rotation."""
    result = image.copy()
    if rng.random() < 0.6:
        k = rng.choice([3, 5, 7])
        result = cv2.GaussianBlur(result, (k, k), 0)
    if rng.random() < 0.5:
        noise = np.random.normal(0, rng.uniform(10, 30), result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        _, enc = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 70)])
        result = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if rng.random() < 0.5:
        result = np.clip(result.astype(np.float32) * rng.uniform(0.7, 1.3) +
                         rng.randint(-30, 30), 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        h, w = result.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-2, 2), 1.0)
        result = cv2.warpAffine(result, M, (w, h), borderValue=(255, 255, 255))
    return result


def apply_lapsrn(image):
    from core.enhancement import enhance_lapsrn
    return enhance_lapsrn(image)


def apply_realesrgan(image):
    from core.enhancement import enhance_realesrgan
    return enhance_realesrgan(image)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def annotations_to_yolo(annotations, img_w, img_h):
    lines = []
    for franco, x, y, w, h in annotations:
        cid = CLASS_TO_ID[franco]
        lines.append(f"{cid} {(x + w / 2) / img_w:.6f} {(y + h / 2) / img_h:.6f} "
                     f"{w / img_w:.6f} {h / img_h:.6f}")
    return "\n".join(lines)


def scale_annotations(annotations, scale=2):
    return [(f, x * scale, y * scale, w * scale, h * scale)
            for f, x, y, w, h in annotations]


def draw_preview(image, annotations):
    """Draw red bounding boxes + Franco class names."""
    vis = image.copy()
    for franco, x, y, w, h in annotations:
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        fs = 0.5 if vis.shape[1] < 500 else 0.7
        (tw, th), _ = cv2.getTextSize(franco, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(vis, (int(x), int(y) - th - 4), (int(x) + tw, int(y)), (0, 0, 255), -1)
        cv2.putText(vis, franco, (int(x), int(y) - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1)
    return vis


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def collect_existing_dataset():
    """Collect all images + labels from existing dataset."""
    items = []
    img_dir = os.path.join(EXISTING_DATASET, "train", "images")
    lbl_dir = os.path.join(EXISTING_DATASET, "train", "labels")
    if not os.path.exists(img_dir):
        print(f"  No existing dataset found at {img_dir}")
        return items

    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(img_dir, fname)
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        lbl_path = os.path.join(lbl_dir, lbl_name)
        if os.path.exists(lbl_path):
            items.append((img_path, lbl_path))
    return items


def create_dirs():
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            os.makedirs(os.path.join(OUTPUT_DIR, split, sub), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "preview"), exist_ok=True)


def write_data_yaml():
    yaml_path = os.path.join(OUTPUT_DIR, "data.yaml")
    names_str = ", ".join(f"'{n}'" for n in CLASS_NAMES)
    with open(yaml_path, "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"test: test/images\n\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: [{names_str}]\n")
    print(f"Written {yaml_path} ({len(CLASS_NAMES)} classes)")


def _write_item(split, name, img, label_content):
    """Write one image + label to disk immediately."""
    img_out = os.path.join(OUTPUT_DIR, split, "images", f"{name}.png")
    lbl_out = os.path.join(OUTPUT_DIR, split, "labels", f"{name}.txt")
    cv2.imwrite(img_out, img)
    with open(lbl_out, "w") as f:
        f.write(label_content + "\n")


def build_dataset(synthetic_count, char_paths, rng):
    """Combine existing + synthetic data, split into train/val/test.
    Writes each item to disk immediately to avoid memory issues."""
    create_dirs()
    write_data_yaml()

    # --- Pre-compute total count and split assignments ---
    print("\n--- Collecting existing dataset ---")
    existing = collect_existing_dataset()
    n_existing = len(existing)
    n_synthetic = synthetic_count * 4  # 4 variants each
    total = n_existing + n_synthetic
    print(f"  Existing: {n_existing} images")
    print(f"  Synthetic: {synthetic_count} composites x4 = {n_synthetic} images")
    print(f"  Total: {total} images")

    # Pre-assign splits using shuffled indices
    indices = list(range(total))
    rng.shuffle(indices)
    n_train = int(total * 0.7)
    n_val = int(total * 0.2)
    splits = [""] * total
    for i in indices[:n_train]:
        splits[i] = "train"
    for i in indices[n_train:n_train + n_val]:
        splits[i] = "val"
    for i in indices[n_train + n_val:]:
        splits[i] = "test"

    split_counts = {"train": 0, "val": 0, "test": 0}
    preview_count = 0

    # --- Phase 1: Copy existing dataset (no memory overhead) ---
    print("\n--- Copying existing dataset ---")
    for i, (img_path, lbl_path) in enumerate(existing):
        name = f"real_{i:05d}"
        split = splits[i]
        img_out = os.path.join(OUTPUT_DIR, split, "images", f"{name}.jpg")
        lbl_out = os.path.join(OUTPUT_DIR, split, "labels", f"{name}.txt")
        shutil.copy2(img_path, img_out)
        shutil.copy2(lbl_path, lbl_out)
        split_counts[split] += 1
        if (i + 1) % 500 == 0:
            print(f"  Copied {i + 1}/{n_existing}")

    print(f"  Copied {n_existing} existing images")

    # --- Phase 2: Generate synthetic composites (write immediately) ---
    print(f"\n--- Generating {synthetic_count} synthetic composites ---")
    synth_idx = 0
    for idx in range(synthetic_count):
        canvas, annotations = generate_composite(char_paths, rng)
        if not annotations:
            continue

        base = f"synth_{idx:05d}"
        label_normal = annotations_to_yolo(annotations, canvas.shape[1], canvas.shape[0])
        ann_scaled = scale_annotations(annotations, 2)

        # Normal — write immediately
        i = n_existing + synth_idx
        _write_item(splits[i], f"{base}_normal", canvas, label_normal)
        split_counts[splits[i]] += 1
        if preview_count < 3:
            preview = draw_preview(canvas, annotations)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "preview", f"{base}_normal.png"), preview)
            preview_count += 1
        synth_idx += 1

        # LapSRN
        lapsrn_img = apply_lapsrn(canvas)
        lapsrn_label = annotations_to_yolo(ann_scaled, lapsrn_img.shape[1], lapsrn_img.shape[0])
        i = n_existing + synth_idx
        _write_item(splits[i], f"{base}_lapsrn", lapsrn_img, lapsrn_label)
        split_counts[splits[i]] += 1
        synth_idx += 1
        del lapsrn_img

        # Real-ESRGAN
        esrgan_img = apply_realesrgan(canvas)
        esrgan_label = annotations_to_yolo(ann_scaled, esrgan_img.shape[1], esrgan_img.shape[0])
        i = n_existing + synth_idx
        _write_item(splits[i], f"{base}_realesrgan", esrgan_img, esrgan_label)
        split_counts[splits[i]] += 1
        synth_idx += 1
        del esrgan_img

        # Augmented
        aug_img = apply_augmentation(canvas, rng)
        i = n_existing + synth_idx
        _write_item(splits[i], f"{base}_augmented", aug_img, label_normal)
        split_counts[splits[i]] += 1
        synth_idx += 1
        del aug_img, canvas

        if (idx + 1) % 50 == 0 or idx < 5:
            print(f"  [{idx + 1}/{synthetic_count}] done "
                  f"(train:{split_counts['train']} val:{split_counts['val']} test:{split_counts['test']})")

    # --- Phase 3: Generate preview images with bounding boxes ---
    print("\n--- Generating preview images ---")
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(OUTPUT_DIR, split, "images")
        lbl_dir = os.path.join(OUTPUT_DIR, split, "labels")
        if not os.path.exists(img_dir):
            continue
        for fname in sorted(os.listdir(img_dir))[:3]:
            if not fname.lower().endswith((".png", ".jpg")):
                continue
            img = cv2.imread(os.path.join(img_dir, fname))
            lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")
            if img is None or not os.path.exists(lbl_path):
                continue
            h, w = img.shape[:2]
            anns = []
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cid, xc, yc, bw, bh = int(parts[0]), *[float(p) for p in parts[1:]]
                    franco = CLASS_NAMES[cid]
                    anns.append((franco, int((xc - bw / 2) * w), int((yc - bh / 2) * h),
                                 int(bw * w), int(bh * h)))
            preview = draw_preview(img, anns)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "preview", f"preview_{split}_{fname}"), preview)

    print(f"\n=== Done! ===")
    print(f"  Train: {split_counts['train']} images")
    print(f"  Val:   {split_counts['val']} images")
    print(f"  Test:  {split_counts['test']} images")
    print(f"  Total: {sum(split_counts.values())} images")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Preview: {OUTPUT_DIR}/preview/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build final YOLO26m character dataset")
    parser.add_argument("--samples", type=int, help="Generate N synthetic composites only (preview mode)")
    parser.add_argument("--count", type=int, default=1500, help="Number of synthetic composites (default: 1500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Indexing EALPR character images...")
    char_paths = index_characters(CHARS_DIR)
    total = sum(len(v) for v in char_paths.values())
    print(f"Found {total} character images across {len(char_paths)} classes")

    count = args.samples if args.samples else args.count
    build_dataset(count, char_paths, rng)


if __name__ == "__main__":
    main()
