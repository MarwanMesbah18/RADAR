#!/usr/bin/env python3
"""Generate clean training charts from results.csv, skipping first 3 warmup epochs."""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import numpy as np

WARMUP_EPOCHS = 3
CHARTS_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# Color palette - matching the presentation theme
PRIMARY = '#1E3A5F'      # Deep navy
SECONDARY = '#2E86AB'    # Steel blue
ACCENT = '#E8553D'       # Coral/red
SUCCESS = '#28A745'       # Green
WARN = '#FFC107'          # Yellow/gold
LIGHT_BG = '#F8F9FA'
GRID_COLOR = '#E0E0E0'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': LIGHT_BG,
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.color': GRID_COLOR,
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'figure.dpi': 200,
})


def style_ax(ax, title, xlabel='Epoch', ylabel=''):
    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def generate_charts(csv_path, model_name, prefix):
    """Generate all training charts for a model."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Skip warmup epochs
    df = df[df['epoch'] > WARMUP_EPOCHS].copy()

    epochs = df['epoch'].values

    # --- 1. Loss Curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['train/box_loss'], color=PRIMARY, linewidth=2.2, label='Box Loss (Train)')
    ax.plot(epochs, df['train/cls_loss'], color=ACCENT, linewidth=2.2, label='Cls Loss (Train)')
    ax.plot(epochs, df['train/dfl_loss'], color=SUCCESS, linewidth=2.2, label='DFL Loss (Train)')
    ax.plot(epochs, df['val/box_loss'], color=PRIMARY, linewidth=2, linestyle='--', label='Box Loss (Val)')
    ax.plot(epochs, df['val/cls_loss'], color=ACCENT, linewidth=2, linestyle='--', label='Cls Loss (Val)')
    ax.plot(epochs, df['val/dfl_loss'], color=SUCCESS, linewidth=2, linestyle='--', label='DFL Loss (Val)')
    style_ax(ax, f'{model_name} — Training & Validation Loss', ylabel='Loss')
    ax.legend(loc='upper right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f'{prefix}_loss.png'))
    plt.close(fig)

    # --- 2. Precision & Recall ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['metrics/precision(B)'], color=PRIMARY, linewidth=2.5, label='Precision')
    ax.plot(epochs, df['metrics/recall(B)'], color=ACCENT, linewidth=2.5, label='Recall')
    ax.fill_between(epochs, df['metrics/precision(B)'], alpha=0.1, color=PRIMARY)
    ax.fill_between(epochs, df['metrics/recall(B)'], alpha=0.1, color=ACCENT)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    style_ax(ax, f'{model_name} — Precision & Recall', ylabel='Score')
    ax.legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f'{prefix}_precision_recall.png'))
    plt.close(fig)

    # --- 3. mAP Curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, df['metrics/mAP50(B)'], color=SECONDARY, linewidth=2.5, label='mAP@50')
    ax.plot(epochs, df['metrics/mAP50-95(B)'], color=ACCENT, linewidth=2.5, label='mAP@50-95')
    ax.fill_between(epochs, df['metrics/mAP50(B)'], alpha=0.15, color=SECONDARY)
    ax.fill_between(epochs, df['metrics/mAP50-95(B)'], alpha=0.15, color=ACCENT)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    style_ax(ax, f'{model_name} — mAP Score', ylabel='mAP')
    ax.legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f'{prefix}_map.png'))
    plt.close(fig)

    # --- 4. Combined overview (2x2) ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: All losses
    ax = axes[0, 0]
    ax.plot(epochs, df['train/box_loss'], color=PRIMARY, linewidth=2, label='Box')
    ax.plot(epochs, df['train/cls_loss'], color=ACCENT, linewidth=2, label='Cls')
    ax.plot(epochs, df['train/dfl_loss'], color=SUCCESS, linewidth=2, label='DFL')
    style_ax(ax, 'Train Loss', ylabel='Loss')
    ax.legend(fontsize=10)

    # Top-right: Precision/Recall
    ax = axes[0, 1]
    ax.plot(epochs, df['metrics/precision(B)'], color=PRIMARY, linewidth=2.5, label='Precision')
    ax.plot(epochs, df['metrics/recall(B)'], color=ACCENT, linewidth=2.5, label='Recall')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    style_ax(ax, 'Precision & Recall')
    ax.legend(fontsize=10)

    # Bottom-left: mAP
    ax = axes[1, 0]
    ax.plot(epochs, df['metrics/mAP50(B)'], color=SECONDARY, linewidth=2.5, label='mAP@50')
    ax.plot(epochs, df['metrics/mAP50-95(B)'], color=ACCENT, linewidth=2.5, label='mAP@50-95')
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    style_ax(ax, 'mAP Scores')
    ax.legend(fontsize=10)

    # Bottom-right: Val losses
    ax = axes[1, 1]
    ax.plot(epochs, df['val/box_loss'], color=PRIMARY, linewidth=2, label='Box')
    ax.plot(epochs, df['val/cls_loss'], color=ACCENT, linewidth=2, label='Cls')
    ax.plot(epochs, df['val/dfl_loss'], color=SUCCESS, linewidth=2, label='DFL')
    style_ax(ax, 'Validation Loss', ylabel='Loss')
    ax.legend(fontsize=10)

    fig.suptitle(f'{model_name} — Training Overview (Epochs {WARMUP_EPOCHS+1}–{int(epochs[-1])})',
                 fontsize=20, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(CHARTS_DIR, f'{prefix}_overview.png'))
    plt.close(fig)

    # --- 5. Best metrics summary ---
    best_map50 = df['metrics/mAP50(B)'].max()
    best_map5095 = df['metrics/mAP50-95(B)'].max()
    best_prec = df['metrics/precision(B)'].max()
    best_recall = df['metrics/recall(B)'].max()
    best_epoch = df.loc[df['metrics/mAP50(B)'].idxmax(), 'epoch']

    print(f"\n{'='*50}")
    print(f"  {model_name} — Best Metrics (after warmup)")
    print(f"{'='*50}")
    print(f"  Best mAP@50:    {best_map50:.4f} (epoch {int(best_epoch)})")
    print(f"  Best mAP@50-95: {best_map5095:.4f}")
    print(f"  Best Precision:  {best_prec:.4f}")
    print(f"  Best Recall:     {best_recall:.4f}")
    print(f"{'='*50}\n")

    return {
        'best_map50': best_map50,
        'best_map5095': best_map5095,
        'best_prec': best_prec,
        'best_recall': best_recall,
        'best_epoch': int(best_epoch),
        'total_epochs': int(epochs[-1]),
    }


if __name__ == '__main__':
    # OCR V2 training results
    ocr_metrics = generate_charts(
        '/home/mesbah/Desktop/Projects/RADAR/output/Train/results.csv',
        'OCR V2 (YOLO26m)',
        'ocr_v2'
    )

    # Seatbelt training results
    seatbelt_metrics = generate_charts(
        '/home/mesbah/Desktop/Projects/RADAR/output/Train_seatbelt/seatbelt_v1_all_outputs/results.csv',
        'Seatbelt+Mobile (YOLOv11m)',
        'seatbelt'
    )

    print("\nCharts saved to:", CHARTS_DIR)
    print("OCR V2 metrics:", ocr_metrics)
    print("Seatbelt metrics:", seatbelt_metrics)
