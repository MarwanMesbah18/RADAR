"""
Boost weak classes by augmenting existing real images that contain them.

Finds images with weak class instances in the train split, creates
augmented copies (rotation, blur, noise, brightness, contrast), and
appends them to the dataset. Does NOT modify or delete existing data.

Usage:
  python scripts/boost_weak_classes.py               # full boost
  python scripts/boost_weak_classes.py --samples 10   # preview 10
"""

import argparse
import os
import random
import sys
import glob
import shutil

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "Datasets", "characters_final")

CLASS_NAMES = [
    '0', '1', '2', '3', '4', '5', '6', '7', '7aa', '8', '9',
    'Taa', 'Thaa', 'ain', 'alif', 'baa', 'daad', 'daal', 'faa',
    'ghayn', 'haa', 'jeem', 'kaaf', 'khaa', 'laam', 'meem', 'noon',
    'qaaf', 'raa', 'saad', 'seen', 'sheen', 'taa', 'thaa', 'waw',
    'yaa', 'zaal', 'zay'
]

WEAK_CLASSES = ['daad', 'zaal', 'kaaf', 'taa', '7aa', 'zay',
                'ghayn', 'khaa', 'Thaa', 'sheen', 'thaa']
WEAK_IDS = {CLASS_NAMES.index(c) for c in WEAK_CLASSES}
TARGET = 600  # target instances per weak class


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_instances(lbl_dir, weak_ids):
    """Count weak class instances in label files."""
    counts = {}
    for cid in weak_ids:
        counts[cid] = 0
    for lbl_file in glob.glob(os.path.join(lbl_dir, "*.txt")):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    if cid in weak_ids:
                        counts[cid] = counts.get(cid, 0) + 1
    return counts


def find_weak_images(img_dir, lbl_dir, weak_ids):
    """Find image+label pairs that contain at least one weak class."""
    pairs = []
    for lbl_file in sorted(glob.glob(os.path.join(lbl_dir, "*.txt"))):
        has_weak = False
        with open(lbl_file) as f:
            for line in f:
                cid = int(line.strip().split()[0])
                if cid in weak_ids:
                    has_weak = True
                    break
        if has_weak:
            lbl_name = os.path.basename(lbl_file)
            stem = os.path.splitext(lbl_name)[0]
            # Find matching image
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = os.path.join(img_dir, stem + ext)
                if os.path.exists(img_path):
                    pairs.append((img_path, lbl_file))
                    break
    return pairs


def augment_image(img, rng):
    """Apply random augmentation to an image. Returns augmented image."""
    result = img.copy()

    # Rotation (±5 degrees)
    if rng.random() < 0.7:
        angle = rng.uniform(-5, 5)
        h, w = result.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        result = cv2.warpAffine(result, M, (w, h),
                                borderValue=(255, 255, 255) if result.mean() > 128 else (0, 0, 0))

    # Gaussian blur
    if rng.random() < 0.5:
        k = rng.choice([3, 5])
        result = cv2.GaussianBlur(result, (k, k), 0)

    # Gaussian noise
    if rng.random() < 0.5:
        sigma = rng.uniform(5, 20)
        noise = np.random.normal(0, sigma, result.shape).astype(np.float32)
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Brightness/contrast
    if rng.random() < 0.6:
        alpha = rng.uniform(0.75, 1.25)
        beta = rng.randint(-25, 25)
        result = np.clip(result.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # JPEG compression
    if rng.random() < 0.3:
        q = rng.randint(40, 75)
        _, enc = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, q])
        result = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    # Horizontal scale (squeeze/stretch slightly)
    if rng.random() < 0.3:
        h, w = result.shape[:2]
        scale_x = rng.uniform(0.9, 1.1)
        new_w = max(1, int(w * scale_x))
        result = cv2.resize(result, (new_w, h))
        # Pad back to original size
        if new_w < w:
            pad_left = (w - new_w) // 2
            pad_right = w - new_w - pad_left
            result = cv2.copyMakeBorder(result, 0, 0, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=[255, 255, 255])
        elif new_w > w:
            start = (new_w - w) // 2
            result = result[:, start:start + w]

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Boost weak classes via augmentation")
    parser.add_argument("--samples", type=int, help="Preview: only generate N augmented copies")
    parser.add_argument("--augments-per-image", type=int, default=3,
                        help="Number of augmented copies per source image (default: 3)")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Count current instances across all splits
    print("=== Current weak class counts ===")
    total_counts = {}
    for split in ["train", "val", "test"]:
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        if os.path.exists(lbl_dir):
            split_counts = count_instances(lbl_dir, WEAK_IDS)
            for cid, cnt in split_counts.items():
                total_counts[cid] = total_counts.get(cid, 0) + cnt

    for cid in sorted(WEAK_IDS):
        name = CLASS_NAMES[cid]
        c = total_counts.get(cid, 0)
        need = max(0, TARGET - c)
        print(f"  {name:>10}: {c:>5} current, need {need:>4} more")

    total_needed = sum(max(0, TARGET - total_counts.get(cid, 0)) for cid in WEAK_IDS)
    print(f"  Total instances needed: {total_needed}")

    # Find source images with weak classes in train split
    train_img_dir = os.path.join(DATASET_DIR, "train", "images")
    train_lbl_dir = os.path.join(DATASET_DIR, "train", "labels")
    source_pairs = find_weak_images(train_img_dir, train_lbl_dir, WEAK_IDS)
    print(f"\n  Source images with weak classes: {len(source_pairs)}")

    if not source_pairs:
        print("ERROR: No source images found!")
        return

    # Calculate how many augmented copies we need
    # Each source image contains ~1-3 weak instances, each augment adds them again
    weak_per_image = total_needed // len(source_pairs) + 1
    augs_per_img = min(weak_per_image, args.augments_per_image)
    if args.samples:
        total_to_gen = args.samples
    else:
        total_to_gen = len(source_pairs) * augs_per_img

    print(f"  Augmented copies per source: {augs_per_img}")
    print(f"  Total augmented images to generate: {total_to_gen}")

    # Pre-assign splits (70/20/10)
    indices = list(range(total_to_gen))
    rng.shuffle(indices)
    n_train = int(total_to_gen * 0.7)
    n_val = int(total_to_gen * 0.2)
    splits = [""] * total_to_gen
    for i in indices[:n_train]:
        splits[i] = "train"
    for i in indices[n_train:n_train + n_val]:
        splits[i] = "val"
    for i in indices[n_train + n_val:]:
        splits[i] = "test"

    # Find highest existing boost number
    existing_max = 0
    for split in ["train", "val", "test"]:
        for f in glob.glob(os.path.join(DATASET_DIR, split, "images", "boost_*")):
            num = int(os.path.basename(f).split("_")[1].split("_")[0])
            existing_max = max(existing_max, num)
    start_idx = existing_max + 1

    # Generate augmented copies
    print(f"\n--- Generating {total_to_gen} augmented images (starting idx {start_idx}) ---")
    split_counts = {"train": 0, "val": 0, "test": 0}
    generated = 0

    # Shuffle source pairs and cycle through them
    source_cycle = source_pairs.copy()
    rng.shuffle(source_cycle)

    for i in range(total_to_gen):
        src_img_path, src_lbl_path = source_cycle[i % len(source_cycle)]

        # Read and augment image
        img = cv2.imread(src_img_path)
        if img is None:
            continue
        aug_img = augment_image(img, rng)

        # Copy label as-is (same normalized coordinates still apply)
        with open(src_lbl_path) as f:
            label_content = f.read().strip()

        name = f"boost_{start_idx + i:05d}"
        split = splits[i]
        img_ext = os.path.splitext(src_img_path)[1]

        img_out = os.path.join(DATASET_DIR, split, "images", f"{name}{img_ext}")
        lbl_out = os.path.join(DATASET_DIR, split, "labels", f"{name}.txt")

        cv2.imwrite(img_out, aug_img)
        with open(lbl_out, "w") as f:
            f.write(label_content + "\n")

        split_counts[split] += 1
        generated += 1

        if (i + 1) % 200 == 0 or i < 3:
            print(f"  [{i + 1}/{total_to_gen}] done "
                  f"(train:+{split_counts['train']} val:+{split_counts['val']} test:+{split_counts['test']})")

    print(f"\n=== Done! Added {generated} augmented images ===")
    print(f"  Train: +{split_counts['train']}")
    print(f"  Val:   +{split_counts['val']}")
    print(f"  Test:  +{split_counts['test']}")

    # Recount
    print(f"\n=== Updated weak class counts ===")
    new_total = {}
    for split in ["train", "val", "test"]:
        lbl_dir = os.path.join(DATASET_DIR, split, "labels")
        if os.path.exists(lbl_dir):
            c = count_instances(lbl_dir, WEAK_IDS)
            for cid, cnt in c.items():
                new_total[cid] = new_total.get(cid, 0) + cnt

    for cid in sorted(WEAK_IDS):
        name = CLASS_NAMES[cid]
        old = total_counts.get(cid, 0)
        new = new_total.get(cid, 0)
        print(f"  {name:>10}: {old:>5} → {new:>5} (+{new - old})")


if __name__ == "__main__":
    main()
