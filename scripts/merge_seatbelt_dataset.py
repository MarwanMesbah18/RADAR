#!/usr/bin/env python3
"""Merge seatbelt datasets: v3 base + v2 mobile labels + v5 unique images.

Produces seatbelt_merged/ with 5 classes:
  0: person-noseatbelt
  1: person-seatbelt
  2: seatbelt
  3: windshield
  4: mobile
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

BASE = Path("/home/mesbah/Desktop/Projects/RADAR/Datasets/Seatbelts_Data")
V3 = BASE / "Seatbelt Detection.v3i.yolov11"
V2 = BASE / "Seatbelt Detection.v2i.yolov11"
V5 = BASE / "Seatbelt Detection.v5i.yolov11"
MERGED = BASE / "seatbelt_merged"

SPLITS = ["train", "valid", "test"]


def source_prefix(filename: str) -> str:
    """Extract Roboflow source image prefix (everything before _jpg.rf. or .rf.)."""
    name = Path(filename).stem
    if "_jpg.rf." in filename:
        return name.split("_jpg.rf.")[0]
    parts = name.split(".rf.")
    return parts[0] if len(parts) > 1 else name


def build_prefix_index(dataset_dir: Path) -> dict[str, dict]:
    """Build {prefix: {split, img_path, label_path}} for ALL splits combined."""
    index = {}
    for split in SPLITS:
        images_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        if not images_dir.exists():
            continue
        for f in images_dir.iterdir():
            if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            prefix = source_prefix(f.name)
            label_file = labels_dir / (f.stem + ".txt")
            index[prefix] = {
                "split": split,
                "img_path": f,
                "label_path": label_file,
            }
    return index


def step0_delete_old():
    """Step 0: Delete old merged dataset."""
    print("\n=== Step 0: Delete old merged dataset ===")
    if MERGED.exists():
        shutil.rmtree(MERGED)
        print(f"  Deleted {MERGED}")
    else:
        print(f"  {MERGED} does not exist, skipping")


def step1_copy_v3_base():
    """Step 1: Copy v3 as the base dataset (640x640)."""
    print("\n=== Step 1: Copy v3 as base (640x640) ===")
    for split in SPLITS:
        for sub in ["images", "labels"]:
            src = V3 / split / sub
            dst = MERGED / split / sub
            shutil.copytree(src, dst)
            count = len(list(dst.iterdir()))
            print(f"  {split}/{sub}: copied {count} files")

    total = sum(len(list((MERGED / s / "images").iterdir())) for s in SPLITS)
    print(f"  Total base images: {total}")


def step2_inject_mobile_from_v2():
    """Step 2: Inject mobile labels from v2 into matching v3 images.

    Uses a GLOBAL prefix index for both v2 and v3 (searches ALL splits).
    This is the fix for the previous attempt's low mobile count.
    """
    print("\n=== Step 2: Inject mobile labels from v2 into v3 matches ===")

    # Build global indexes for v2 and v3
    v2_index = build_prefix_index(V2)
    v3_index = build_prefix_index(V3)

    # Build merged label index: prefix -> label_path (from v3 base)
    merged_labels = {}
    for split in SPLITS:
        labels_dir = MERGED / split / "labels"
        for lf in labels_dir.iterdir():
            if lf.suffix == ".txt":
                merged_labels[source_prefix(lf.name)] = lf

    injected_files = 0
    injected_labels = 0

    # Find all v2 source images that have mobile labels
    for v2_prefix, v2_info in v2_index.items():
        v2_label = v2_info["label_path"]
        if not v2_label.exists():
            continue

        v2_text = v2_label.read_text().strip()
        if not v2_text:
            continue

        # Extract mobile lines (class 0 in v2 = mobile)
        mobile_lines = [line.strip() for line in v2_text.split("\n") if line.strip().startswith("0 ")]
        if not mobile_lines:
            continue

        # Check if this source exists in v3 (via merged labels copied from v3)
        if v2_prefix not in merged_labels:
            continue

        merged_label = merged_labels[v2_prefix]

        # Remap mobile lines: class 0 (v2) → class 4 (merged)
        remapped = []
        for line in mobile_lines:
            parts = line.split()
            parts[0] = "4"
            remapped.append(" ".join(parts))

        # Append to merged label
        existing = merged_label.read_text().strip()
        if existing:
            merged_label.write_text(existing + "\n" + "\n".join(remapped) + "\n")
        else:
            merged_label.write_text("\n".join(remapped) + "\n")

        injected_files += 1
        injected_labels += len(remapped)

    print(f"  Images with mobile injected: {injected_files}")
    print(f"  Mobile labels injected: {injected_labels}")


def step3_add_v2_only_mobile_images():
    """Step 3: Add v2-only images (source with mobile NOT in v3) with remapped labels."""
    print("\n=== Step 3: Add v2-only images with mobile class ===")

    # Remap: v2 class → merged class (-1 = drop)
    # v2: 0:mobile, 1:person, 2:person-noseatbelt, 3:person-seatbelt, 4:seatbelt, 5:windshield
    remap = {0: 4, 1: -1, 2: 0, 3: 1, 4: 2, 5: 3}

    # Build global index of all merged prefixes (from v3 base)
    merged_prefixes = set()
    for split in SPLITS:
        images_dir = MERGED / split / "images"
        for f in images_dir.iterdir():
            merged_prefixes.add(source_prefix(f.name))

    # Build v2 global index
    v2_index = build_prefix_index(V2)

    added = 0
    mobile_labels = 0

    for v2_prefix, v2_info in v2_index.items():
        if v2_prefix in merged_prefixes:
            continue  # Already in v3, handled in step 2

        v2_img = v2_info["img_path"]
        v2_label = v2_info["label_path"]

        # Copy image to merged train
        dst_img = MERGED / "train" / "images" / v2_img.name
        shutil.copy2(v2_img, dst_img)

        # Remap label
        dst_label = MERGED / "train" / "labels" / (v2_img.stem + ".txt")

        if v2_label.exists():
            lines = v2_label.read_text().strip().split("\n")
            remapped = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                old_class = int(parts[0])
                new_class = remap.get(old_class, -1)
                if new_class == -1:
                    continue
                parts[0] = str(new_class)
                remapped.append(" ".join(parts))

            dst_label.write_text("\n".join(remapped) + "\n" if remapped else "")

            for line in remapped:
                if line.startswith("4 "):
                    mobile_labels += 1
        else:
            dst_label.touch()

        added += 1
        merged_prefixes.add(v2_prefix)

    print(f"  v2-only images added: {added}")
    print(f"  Mobile labels in v2-only images: {mobile_labels}")


def step4_add_v5_unique_images():
    """Step 4: Add v5 images whose source is NOT in v3."""
    print("\n=== Step 4: Add v5 unique images (not in v3) ===")

    # Build global index of all merged prefixes (v3 base + v2-only)
    merged_prefixes = set()
    for split in SPLITS:
        images_dir = MERGED / split / "images"
        for f in images_dir.iterdir():
            merged_prefixes.add(source_prefix(f.name))

    # Build v5 global index
    v5_index = build_prefix_index(V5)

    added = 0

    for v5_prefix, v5_info in v5_index.items():
        if v5_prefix in merged_prefixes:
            continue  # Already in v3

        v5_img = v5_info["img_path"]
        v5_label = v5_info["label_path"]

        # Copy image to merged train
        dst_img = MERGED / "train" / "images" / v5_img.name
        shutil.copy2(v5_img, dst_img)

        # Copy label (no remap — v5 uses same class IDs as v3: 0,1,2,3)
        dst_label = MERGED / "train" / "labels" / (v5_img.stem + ".txt")
        if v5_label.exists():
            shutil.copy2(v5_label, dst_label)
        else:
            dst_label.touch()

        added += 1
        merged_prefixes.add(v5_prefix)

    print(f"  v5 unique images added: {added}")


def step5_create_data_yaml():
    """Step 5: Create data.yaml for the merged dataset."""
    print("\n=== Step 5: Create data.yaml ===")
    yaml_content = f"""path: {MERGED}
train: train/images
val: valid/images
test: test/images

nc: 5
names:
  0: person-noseatbelt
  1: person-seatbelt
  2: seatbelt
  3: windshield
  4: mobile
"""
    yaml_path = MERGED / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  Written to {yaml_path}")


def print_summary():
    """Print final dataset summary."""
    print("\n=== Final Summary ===")
    for split in SPLITS:
        imgs = len(list((MERGED / split / "images").iterdir()))
        labels = len(list((MERGED / split / "labels").iterdir()))
        print(f"  {split}: {imgs} images, {labels} labels")

    total = sum(len(list((MERGED / s / "images").iterdir())) for s in SPLITS)
    print(f"  TOTAL: {total} images")


if __name__ == "__main__":
    step0_delete_old()
    step1_copy_v3_base()
    step2_inject_mobile_from_v2()
    step3_add_v2_only_mobile_images()
    step4_add_v5_unique_images()
    step5_create_data_yaml()
    print_summary()
    print("\nMerge complete!")
