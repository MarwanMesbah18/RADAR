#!/usr/bin/env python3
"""Validate merged seatbelt dataset and generate annotated sample images.

Checks: class distribution, image sizes, label integrity, duplicates.
Generates: 10 annotated sample images in seatbelt_merged/samples/
"""

import os
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import random

MERGED = Path("/home/mesbah/Desktop/Projects/RADAR/Datasets/Seatbelts_Data/seatbelt_merged")
SPLITS = ["train", "valid", "test"]

CLASS_NAMES = {
    0: "person-noseatbelt",
    1: "person-seatbelt",
    2: "seatbelt",
    3: "windshield",
    4: "mobile",
}

CLASS_COLORS = {
    0: (255, 0, 0),       # RED
    1: (0, 200, 0),       # GREEN
    2: (0, 100, 255),     # BLUE
    3: (255, 255, 0),     # YELLOW
    4: (255, 0, 255),     # MAGENTA
}


def check_class_distribution():
    """6a: Count all labels per class across all splits."""
    print("\n=== 6a: Class Distribution Audit ===")
    total_counts = defaultdict(int)
    split_counts = {s: defaultdict(int) for s in SPLITS}
    stray_classes = set()

    for split in SPLITS:
        labels_dir = MERGED / split / "labels"
        for label_file in sorted(labels_dir.iterdir()):
            if label_file.suffix != ".txt":
                continue
            text = label_file.read_text().strip()
            if not text:
                continue
            for line in text.split("\n"):
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls > 4:
                    stray_classes.add(cls)
                split_counts[split][cls] += 1
                total_counts[cls] += 1

    for cls in sorted(total_counts.keys()):
        name = CLASS_NAMES.get(cls, f"UNKNOWN({cls})")
        per_split = " | ".join(f"{s}: {split_counts[s][cls]}" for s in SPLITS)
        print(f"  Class {cls} ({name}): {total_counts[cls]} total ({per_split})")

    print(f"\n  All 5 classes present: {set(total_counts.keys()) == {0, 1, 2, 3, 4}}")
    if stray_classes:
        print(f"  WARNING: Stray class IDs found: {stray_classes}")
    else:
        print("  No stray class IDs (all 0-4)")

    return total_counts


def check_image_sizes():
    """6b: Verify all images are 640x640."""
    print("\n=== 6b: Image Size Check ===")
    bad_sizes = defaultdict(int)
    total = 0

    for split in SPLITS:
        images_dir = MERGED / split / "images"
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            total += 1
            try:
                with Image.open(img_file) as img:
                    w, h = img.size
                    if (w, h) != (640, 640):
                        bad_sizes[(w, h)] += 1
            except Exception as e:
                print(f"  CORRUPT: {img_file.name}: {e}")

    print(f"  Total images checked: {total}")
    if bad_sizes:
        print(f"  WARNING: Non-640x640 images found:")
        for size, count in sorted(bad_sizes.items()):
            print(f"    {size[0]}x{size[1]}: {count} images")
    else:
        print("  All images are 640x640 ✓")

    return bad_sizes


def check_label_integrity():
    """6c: Check label files exist, aren't empty, have valid coordinates."""
    print("\n=== 6c: Label Integrity Check ===")
    missing_labels = 0
    empty_labels = 0
    invalid_coords = 0
    total = 0

    for split in SPLITS:
        images_dir = MERGED / split / "images"
        labels_dir = MERGED / split / "labels"

        for img_file in sorted(images_dir.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            total += 1

            label_file = labels_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                missing_labels += 1
                continue

            text = label_file.read_text().strip()
            if not text:
                empty_labels += 1
                continue

            for line in text.split("\n"):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"  MALFORMED LINE in {label_file.name}: '{line}'")
                    continue
                try:
                    coords = [float(x) for x in parts[1:]]
                    if any(c < 0 or c > 1 for c in coords):
                        invalid_coords += 1
                        print(f"  INVALID COORDS in {label_file.name}: {coords}")
                except ValueError:
                    print(f"  PARSE ERROR in {label_file.name}: '{line}'")

    print(f"  Total images: {total}")
    print(f"  Missing labels: {missing_labels}")
    print(f"  Empty labels: {empty_labels}")
    print(f"  Invalid coordinates: {invalid_coords}")

    if missing_labels == 0 and empty_labels == 0 and invalid_coords == 0:
        print("  All labels valid ✓")
    return missing_labels, empty_labels, invalid_coords


def check_duplicates():
    """6d: Check for duplicate filenames and cross-split leakage."""
    print("\n=== 6d: Duplicate Check ===")

    # Check within each split
    for split in SPLITS:
        names = [f.name for f in (MERGED / split / "images").iterdir()]
        dupes = len(names) - len(set(names))
        if dupes:
            print(f"  {split}: {dupes} duplicate filenames!")
        else:
            print(f"  {split}: no duplicates ✓")

    # Check cross-split leakage
    all_names = {}
    for split in SPLITS:
        for f in (MERGED / split / "images").iterdir():
            name = f.name
            if name in all_names:
                print(f"  LEAKAGE: {name} in both {all_names[name]} and {split}")
            all_names[name] = split

    print(f"  Total unique filenames: {len(all_names)}")


def generate_samples():
    """6e: Generate 10 annotated sample images covering all 5 classes."""
    print("\n=== 6e: Generating 10 Annotated Sample Images ===")
    samples_dir = MERGED / "samples"
    samples_dir.mkdir(exist_ok=True)

    # Find images containing each class
    class_images = defaultdict(list)  # class_id -> [(split, img_path, label_path)]

    for split in SPLITS:
        images_dir = MERGED / split / "images"
        labels_dir = MERGED / split / "labels"

        for img_file in sorted(images_dir.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            label_file = labels_dir / (img_file.stem + ".txt")
            if not label_file.exists():
                continue

            text = label_file.read_text().strip()
            if not text:
                continue

            classes_in_file = set()
            for line in text.split("\n"):
                parts = line.strip().split()
                if parts:
                    classes_in_file.add(int(parts[0]))

            for cls in classes_in_file:
                class_images[cls].append((split, img_file, label_file))

    # Print available counts
    for cls in sorted(class_images.keys()):
        print(f"  Class {cls} ({CLASS_NAMES[cls]}): {len(class_images[cls])} images available")

    # Pick 2 images per class, preferring images with fewer classes (cleaner samples)
    selected = []
    for cls in range(5):
        available = class_images.get(cls, [])
        if not available:
            print(f"  WARNING: No images found for class {cls} ({CLASS_NAMES[cls]})!")
            continue

        # Sort by number of classes in image (prefer simpler images)
        scored = []
        for entry in available:
            split, img_path, label_path = entry
            text = label_path.read_text().strip()
            n_classes = len(set(int(l.split()[0]) for l in text.split("\n") if l.strip()))
            scored.append((n_classes, entry))

        scored.sort(key=lambda x: x[0])
        # Pick 2 from the simpler half
        pool = [e for _, e in scored[:max(len(scored) // 2, 2)]]
        picks = random.sample(pool, min(2, len(pool)))
        selected.extend(picks)

    # Remove duplicates (same image selected for multiple classes)
    seen = set()
    unique_selected = []
    for entry in selected:
        key = entry[1].stem
        if key not in seen:
            seen.add(key)
            unique_selected.append(entry)
    selected = unique_selected

    print(f"  Selected {len(selected)} sample images")

    # Draw annotated images
    for i, (split, img_path, label_path) in enumerate(selected):
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        text = label_path.read_text().strip()
        for line in text.split("\n"):
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = [float(x) for x in parts[1:]]

            # Convert YOLO format to pixel coords
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            color = CLASS_COLORS.get(cls, (128, 128, 128))
            name = CLASS_NAMES.get(cls, f"?{cls}")

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            # Draw label background + text
            draw.rectangle([x1, y1 - 14, x1 + len(name) * 7 + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - 13), name, fill=(0, 0, 0))

        out_path = samples_dir / f"sample_{i+1:02d}_{img_path.stem[:30]}.jpg"
        img.save(out_path)
        print(f"  Saved: {out_path.name}")

    print(f"\n  {len(list(samples_dir.iterdir()))} sample images saved to {samples_dir}")


if __name__ == "__main__":
    random.seed(42)
    check_class_distribution()
    check_image_sizes()
    check_label_integrity()
    check_duplicates()
    generate_samples()
    print("\nValidation complete!")
