#!/usr/bin/env python3
"""Fix merged seatbelt dataset:
1. Truncate malformed label lines to 5 values (class x y w h)
2. Resize all images to 640x640
"""

from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

MERGED = Path("/home/mesbah/Desktop/Projects/RADAR/Datasets/Seatbelts_Data/seatbelt_merged")
SPLITS = ["train", "valid", "test"]


def fix_labels():
    """Truncate all label lines to exactly 5 values."""
    print("=== Fixing malformed labels ===")
    fixed = 0

    for split in SPLITS:
        labels_dir = MERGED / split / "labels"
        for label_file in labels_dir.iterdir():
            if label_file.suffix != ".txt":
                continue
            lines = label_file.read_text().strip().split("\n")
            new_lines = []
            changed = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) > 5:
                    new_lines.append(" ".join(parts[:5]))
                    changed = True
                elif len(parts) == 5:
                    new_lines.append(line.strip())
                # Skip empty or malformed lines with < 5 parts
            if changed:
                label_file.write_text("\n".join(new_lines) + "\n")
                fixed += 1

    print(f"  Fixed {fixed} label files")


def resize_image(img_path: Path, target_size=(640, 640)):
    """Resize a single image if not already target size."""
    with Image.open(img_path) as img:
        if img.size == target_size:
            return False
        img_resized = img.resize(target_size, Image.LANCZOS)
        img_resized.save(img_path)
        return True


def resize_all_images():
    """Resize all images to 640x640."""
    print("\n=== Resizing images to 640x640 ===")
    all_images = []
    for split in SPLITS:
        images_dir = MERGED / split / "images"
        for f in images_dir.iterdir():
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                all_images.append(f)

    print(f"  Total images to check: {len(all_images)}")

    resized = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(resize_image, p): p for p in all_images}
        for i, future in enumerate(as_completed(futures), 1):
            if i % 1000 == 0:
                print(f"  Processed {i}/{len(all_images)}...")
            if future.result():
                resized += 1

    print(f"  Resized {resized} images to 640x640")
    print(f"  Already correct: {len(all_images) - resized}")


if __name__ == "__main__":
    fix_labels()
    resize_all_images()
    print("\nFixes applied!")
