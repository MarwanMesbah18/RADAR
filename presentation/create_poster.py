#!/usr/bin/env python3
"""
RADAR A1 Poster — Clean Claude/Anthropic Theme (Fills full A1)
594mm × 841mm at 300 DPI = 7016 × 9933 pixels

Light, minimal, warm. Claude brand palette + embedded training charts.
Content distributed evenly across the ENTIRE A1 canvas.
"""

from PIL import Image, ImageDraw, ImageFont
import os

WIDTH, HEIGHT = 7016, 9933

# ─── Claude / Anthropic Color Palette ───
BG = (252, 250, 245)
BG_SECTION = (247, 244, 238)
TERRACOTTA = (216, 114, 75)
TERRACOTTA_LIGHT = (240, 180, 155)
TERRACOTTA_PALE = (250, 230, 220)
BROWN = (135, 90, 60)
TEXT_PRIMARY = (40, 35, 30)
TEXT_SECONDARY = (100, 90, 80)
TEXT_TERTIARY = (155, 145, 135)
DIVIDER = (220, 215, 208)
CARD_BORDER = (230, 225, 218)

FONT_DIR = "/home/mesbah/.claude/plugins/cache/anthropic-agent-skills/document-skills/f458cee31a75/skills/canvas-design/canvas-fonts/"
CHART_DIR = "/home/mesbah/Desktop/Projects/RADAR/presentation/charts/"
TRAIN_DIR = "/home/mesbah/Desktop/Projects/RADAR/training/output/Train/"
TRAIN_SB_DIR = "/home/mesbah/Desktop/Projects/RADAR/training/output/Train_seatbelt/seatbelt_v1_all_outputs/"

def load_font(name, size):
    try:
        return ImageFont.truetype(FONT_DIR + name, size)
    except:
        return ImageFont.load_default()

# ─── Fonts ───
f_title = load_font("Outfit-Bold.ttf", 480)
f_subtitle = load_font("Jura-Light.ttf", 100)
f_section = load_font("Outfit-Bold.ttf", 64)
f_section_num = load_font("BigShoulders-Bold.ttf", 48)
f_body = load_font("Jura-Light.ttf", 44)
f_body_sm = load_font("Jura-Light.ttf", 38)
f_label = load_font("GeistMono-Regular.ttf", 34)
f_label_sm = load_font("GeistMono-Regular.ttf", 28)
f_label_tiny = load_font("GeistMono-Regular.ttf", 24)
f_tag = load_font("Outfit-Bold.ttf", 36)
f_number = load_font("BigShoulders-Bold.ttf", 80)

img = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(img, "RGBA")

# ─── Layout zones — distribute across full 9933px ───
MARGIN_L = 420
MARGIN_R = WIDTH - 420
CONTENT_W = MARGIN_R - MARGIN_L
COL_2_X = MARGIN_L + CONTENT_W // 2 + 40

# Key Y positions — spread across full canvas
HEADER_TOP = 380
DIVIDER_AFTER_HEADER = 1500

# Section positions spread evenly
S1_Y = 1620       # Overview
S2_Y = 2400       # Pipeline
S3_Y = 3200       # Tech Stack
S4_Y = 3800       # Training Results (Charts)
CHART_ROW2_Y = 5700  # Second chart row
METRICS_Y = 6600     # Key Metrics (moved up to close gap)
FEATURES_Y = 7400    # Key Features
FOOTER_Y = 9400      # Footer

# ═══════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════
draw.rectangle([(MARGIN_L, HEADER_TOP - 20), (MARGIN_L + 120, HEADER_TOP - 10)], fill=(*TERRACOTTA, 255))
draw.text((MARGIN_L, HEADER_TOP), "RADAR", fill=(*TEXT_PRIMARY, 250), font=f_title)
draw.text((MARGIN_L, HEADER_TOP + 540), "Real-Time Vehicle Analysis System", fill=(*TERRACOTTA, 220), font=f_subtitle)
draw.text((MARGIN_L, HEADER_TOP + 660), "Egyptian License Plate Detection  ·  OCR  ·  Violation Detection  ·  2026", fill=(*TEXT_TERTIARY, 180), font=f_body_sm)

# Decorative dots pattern in header area
for dx in range(40):
    for dy in range(3):
        px = WIDTH - 2200 + dx * 30
        py = HEADER_TOP + 100 + dy * 30
        val = (dx * 3 + dy * 7) % 4
        if val < 1:
            draw.ellipse([(px, py), (px + 6, py + 6)], fill=(*TERRACOTTA_LIGHT, 60))

draw.line([(MARGIN_L, DIVIDER_AFTER_HEADER), (MARGIN_R, DIVIDER_AFTER_HEADER)], fill=(*DIVIDER, 200), width=2)


# ═══════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════
draw.rectangle([(MARGIN_L, S1_Y), (MARGIN_L + 6, S1_Y + 50)], fill=(*TERRACOTTA, 255))
draw.text((MARGIN_L + 24, S1_Y - 8), "OVERVIEW", fill=(*TEXT_PRIMARY, 230), font=f_section)
draw.text((MARGIN_L + 380, S1_Y + 6), "01", fill=(*TERRACOTTA_LIGHT, 180), font=f_section_num)

overview_text = [
    "Multi-stage pipeline for Egyptian traffic enforcement",
    "License plate detection, OCR, and super-resolution",
    "Vehicle detection with ByteTrack multi-object tracking",
    "Seatbelt and mobile phone violation detection",
    "Three parallel OCR models with confidence fusion",
    "AI-powered image enhancement for low-quality inputs",
]

for i, line in enumerate(overview_text):
    y = S1_Y + 90 + i * 85
    draw.ellipse([(MARGIN_L + 12, y + 16), (MARGIN_L + 24, y + 28)], fill=(*TERRACOTTA, 180))
    draw.text((MARGIN_L + 44, y), line, fill=(*TEXT_SECONDARY, 200), font=f_body)


# ═══════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════
draw.rectangle([(MARGIN_L, S2_Y), (MARGIN_L + 6, S2_Y + 50)], fill=(*TERRACOTTA, 255))
draw.text((MARGIN_L + 24, S2_Y - 8), "DETECTION PIPELINE", fill=(*TEXT_PRIMARY, 230), font=f_section)
draw.text((MARGIN_L + 750, S2_Y + 6), "02", fill=(*TERRACOTTA_LIGHT, 180), font=f_section_num)

stages = [
    ("Image\nInput", "01"),
    ("Vehicle\nDetection", "02"),
    ("Plate\nLocalization", "03"),
    ("Super\nResolution", "04"),
    ("OCR", "05"),
    ("Results", "06"),
]

blk_w = 920
blk_h = 220
blk_gap = 70
blk_y = S2_Y + 100

for i, (desc, num) in enumerate(stages):
    x = MARGIN_L + i * (blk_w + blk_gap)

    draw.rectangle([(x, blk_y), (x + blk_w, blk_y + blk_h)], fill=(*BG_SECTION, 255), outline=(*CARD_BORDER, 200), width=1)

    cx = x + 40
    cy = blk_y + blk_h // 2
    draw.ellipse([(cx - 24, cy - 24), (cx + 24, cy + 24)], fill=(*TERRACOTTA, 60))
    draw.text((cx - 16, cy - 20), num, fill=(*TERRACOTTA, 200), font=f_label_sm)

    for j, line in enumerate(desc.split("\n")):
        draw.text((cx + 55, blk_y + 45 + j * 60), line, fill=(*TEXT_PRIMARY, 200), font=f_body_sm)

    if i < len(stages) - 1:
        ax = x + blk_w + 10
        ay = blk_y + blk_h // 2
        draw.line([(ax, ay), (ax + blk_gap - 15, ay)], fill=(*TERRACOTTA, 120), width=2)
        draw.polygon([(ax + blk_gap - 15, ay - 10), (ax + blk_gap, ay), (ax + blk_gap - 15, ay + 10)], fill=(*TERRACOTTA, 120))


# ═══════════════════════════════════════════════════
# TECHNOLOGY STACK — Two columns
# ═══════════════════════════════════════════════════
draw.rectangle([(MARGIN_L, S3_Y), (MARGIN_L + 6, S3_Y + 50)], fill=(*TERRACOTTA, 255))
draw.text((MARGIN_L + 24, S3_Y - 8), "TECHNOLOGY STACK", fill=(*TEXT_PRIMARY, 230), font=f_section)
draw.text((MARGIN_L + 700, S3_Y + 6), "03", fill=(*TERRACOTTA_LIGHT, 180), font=f_section_num)

tech_left = [
    ("YOLOv11m", "Plate Detection V1"),
    ("YOLO26m V2", "Character OCR"),
    ("YOLO26m V2-W3", "Weighted OCR V3"),
    ("YOLO26s", "Vehicle Detection"),
]
tech_right = [
    ("ByteTrack", "Multi-Object Tracking"),
    ("LapSRN ×2", "Super-Resolution"),
    ("Real-ESRGAN", "AI Enhancement"),
    ("YOLOv11m", "Seatbelt & Mobile"),
]

col_w = (CONTENT_W - 60) // 2

for col_idx, tech_list in enumerate([tech_left, tech_right]):
    col_x = MARGIN_L + col_idx * (col_w + 60)
    for i, (name, desc) in enumerate(tech_list):
        y = S3_Y + 90 + i * 110
        draw.rectangle([(col_x, y), (col_x + col_w, y + 90)], fill=(*BG_SECTION, 255), outline=(*CARD_BORDER, 150), width=1)
        draw.rectangle([(col_x, y), (col_x + 4, y + 90)], fill=(*TERRACOTTA, 120))
        draw.text((col_x + 20, y + 10), name, fill=(*TERRACOTTA, 220), font=f_label)
        draw.text((col_x + 20, y + 50), desc, fill=(*TEXT_TERTIARY, 180), font=f_label_sm)


# ═══════════════════════════════════════════════════
# TRAINING RESULTS — Charts (LARGE to fill space)
# ═══════════════════════════════════════════════════
draw.rectangle([(MARGIN_L, S4_Y), (MARGIN_L + 6, S4_Y + 50)], fill=(*TERRACOTTA, 255))
draw.text((MARGIN_L + 24, S4_Y - 8), "TRAINING RESULTS", fill=(*TEXT_PRIMARY, 230), font=f_section)
draw.text((MARGIN_L + 620, S4_Y + 6), "04", fill=(*TERRACOTTA_LIGHT, 180), font=f_section_num)

charts_left = [
    (os.path.join(TRAIN_DIR, "training_analysis.png"), "Plate OCR — Training Analysis"),
    (os.path.join(TRAIN_DIR, "per_class_metrics.png"), "Plate OCR — Per-Class Metrics"),
]
charts_right = [
    (os.path.join(TRAIN_SB_DIR, "training_analysis.png"), "Seatbelt — Training Analysis"),
    (os.path.join(TRAIN_SB_DIR, "per_class_metrics.png"), "Seatbelt — Per-Class Metrics"),
]

chart_area_y = S4_Y + 90
# Make charts much taller to fill space
chart_h = 850

def embed_chart(path, x, y, w, h, caption):
    if os.path.exists(path):
        try:
            chart_img = Image.open(path).convert("RGBA")
            aspect = chart_img.width / chart_img.height
            display_w = w - 50
            display_h = int(display_w / aspect)
            if display_h > h - 90:
                display_h = h - 90
                display_w = int(display_h * aspect)
            chart_img = chart_img.resize((display_w, display_h), Image.LANCZOS)
            draw.rectangle([(x, y), (x + w, y + h)], fill=(*BG_SECTION, 255), outline=(*CARD_BORDER, 150), width=1)
            cx = x + (w - display_w) // 2
            cy = y + 20
            img.paste(chart_img, (cx, cy))
            draw.text((x + 24, y + h - 55), caption, fill=(*TEXT_TERTIARY, 160), font=f_label_sm)
        except Exception as e:
            draw.rectangle([(x, y), (x + w, y + h)], fill=(*BG_SECTION, 255), outline=(*CARD_BORDER, 150), width=1)
            draw.text((x + 20, y + 20), f"[Chart: {caption}]", fill=(*TEXT_TERTIARY, 120), font=f_label_sm)
    else:
        draw.rectangle([(x, y), (x + w, y + h)], fill=(*BG_SECTION, 255), outline=(*CARD_BORDER, 150), width=1)
        draw.text((x + 20, y + 20), f"[Not found: {caption}]", fill=(*TEXT_TERTIARY, 120), font=f_label_sm)

# Left column charts
for i, (path, caption) in enumerate(charts_left):
    y = chart_area_y + i * (chart_h + 30)
    embed_chart(path, MARGIN_L, y, col_w, chart_h, caption)

# Right column charts
for i, (path, caption) in enumerate(charts_right):
    y = chart_area_y + i * (chart_h + 30)
    embed_chart(path, COL_2_X, y, col_w, chart_h, caption)


# ═══════════════════════════════════════════════════
# CHARTS ROW 2 — 4 columns
# ═══════════════════════════════════════════════════
extra_charts = [
    (os.path.join(CHART_DIR, "ocr_v2_overview.png"), "OCR V2 — Overview"),
    (os.path.join(CHART_DIR, "seatbelt_overview.png"), "Seatbelt — Overview"),
    (os.path.join(CHART_DIR, "ocr_v2_precision_recall.png"), "OCR V2 — Precision/Recall"),
    (os.path.join(CHART_DIR, "seatbelt_precision_recall.png"), "Seatbelt — Precision/Recall"),
]

chart3_w = (CONTENT_W - 90) // 4

for i, (path, caption) in enumerate(extra_charts):
    x = MARGIN_L + i * (chart3_w + 30)
    embed_chart(path, x, CHART_ROW2_Y, chart3_w, 650, caption)


# ═══════════════════════════════════════════════════
# KEY METRICS — Large numbers
# ═══════════════════════════════════════════════════
draw.rectangle([(MARGIN_L, METRICS_Y), (MARGIN_L + 6, METRICS_Y + 50)], fill=(*TERRACOTTA, 255))
draw.text((MARGIN_L + 24, METRICS_Y - 8), "KEY METRICS", fill=(*TEXT_PRIMARY, 230), font=f_section)
draw.text((MARGIN_L + 480, METRICS_Y + 6), "05", fill=(*TERRACOTTA_LIGHT, 180), font=f_section_num)

metrics = [
    ("3", "OCR Models"),
    ("94%", "Plate mAP"),
    ("91%", "OCR Accuracy"),
    ("25fps", "Detection Speed"),
]

metric_w = (CONTENT_W - 90) // 4

for i, (value, label) in enumerate(metrics):
    x = MARGIN_L + i * (metric_w + 30)
    y = METRICS_Y + 90
    draw.rectangle([(x, y), (x + metric_w, y + 280)], fill=(*TERRACOTTA_PALE, 200), outline=(*TERRACOTTA_LIGHT, 100), width=1)
    draw.text((x + 30, y + 30), value, fill=(*TERRACOTTA, 240), font=f_number)
    draw.text((x + 30, y + 170), label, fill=(*TEXT_SECONDARY, 180), font=f_body_sm)


# ═══════════════════════════════════════════════════
# KEY CAPABILITIES (fills remaining space)
# ═══════════════════════════════════════════════════
draw.rectangle([(MARGIN_L, FEATURES_Y), (MARGIN_L + 6, FEATURES_Y + 50)], fill=(*TERRACOTTA, 255))
draw.text((MARGIN_L + 24, FEATURES_Y - 8), "KEY CAPABILITIES", fill=(*TEXT_PRIMARY, 230), font=f_section)
draw.text((MARGIN_L + 580, FEATURES_Y + 6), "06", fill=(*TERRACOTTA_LIGHT, 180), font=f_section_num)

features = [
    ("Tri-Model OCR", "Three parallel OCR models with confidence-weighted fusion for maximum plate reading accuracy across varied conditions"),
    ("Super Resolution", "AI-powered image enhancement using LapSRN and Real-ESRGAN before OCR on low-quality or distant inputs"),
    ("Vehicle Tracking", "ByteTrack multi-object tracking with persistent identity across frames for real-time traffic monitoring"),
    ("Violation Detection", "Real-time seatbelt and mobile phone usage detection for automated traffic law enforcement"),
]

# Spread features across remaining space
features_area_top = FEATURES_Y + 100
features_area_h = FOOTER_Y - features_area_top - 120
feature_slot_h = features_area_h // len(features)

for i, (ftitle, fdesc) in enumerate(features):
    y = features_area_top + i * feature_slot_h

    # Number
    draw.text((MARGIN_L + 10, y + 5), f"0{i+1}", fill=(*TERRACOTTA, 160), font=f_section_num)

    # Title
    draw.text((MARGIN_L + 140, y + 2), ftitle, fill=(*TEXT_PRIMARY, 220), font=f_section)

    # Description
    draw.text((MARGIN_L + 140, y + 70), fdesc, fill=(*TEXT_SECONDARY, 170), font=f_body_sm)

    # Separator
    if i < len(features) - 1:
        sep_y = y + feature_slot_h - 15
        draw.line([(MARGIN_L + 140, sep_y), (MARGIN_R, sep_y)], fill=(*DIVIDER, 150), width=1)


# ═══════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════
draw.line([(MARGIN_L, FOOTER_Y), (MARGIN_R, FOOTER_Y)], fill=(*DIVIDER, 200), width=1)
draw.text((MARGIN_L, FOOTER_Y + 30), "RADAR  ·  Real-Time Vehicle Analysis  ·  2026", fill=(*TEXT_TERTIARY, 150), font=f_label_sm)
draw.text((MARGIN_R - 700, FOOTER_Y + 30), "A1  ·  594×841mm  ·  300 DPI", fill=(*TEXT_TERTIARY, 100), font=f_label_tiny)


# ═══════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════
output_png = "/home/mesbah/Desktop/Projects/RADAR/presentation/RADAR_poster_A1.png"
output_pdf = "/home/mesbah/Desktop/Projects/RADAR/presentation/RADAR_poster_A1.pdf"

img.save(output_png, "PNG", optimize=True)
print(f"PNG saved: {output_png}")
print(f"Dimensions: {img.size[0]}×{img.size[1]} pixels")

# Also save PDF
rgb = Image.new("RGB", img.size, BG)
rgb.paste(img)
rgb.save(output_pdf, "PDF", resolution=300)
print(f"PDF saved: {output_pdf}")
