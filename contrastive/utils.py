"""
Utility functions for contrastive learning training scripts.
"""

import os
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def setup_environment(rank: int,
                      weights_path: str,
                      pos_dir: str,
                      neg_dir: str):
    """
    Setup environment for contrastive pretraining.
    Checks that weights and data directories exist.
    """
    if rank == 0:
        print("--- Environment Setup ---")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"YOLO11 weights not found: {weights_path}\n"
            f"Download with: from ultralytics import YOLO; YOLO('yolo11n.pt')"
        )
    else:
        if rank == 0:
            print(f"Weights: {weights_path} [OK]")

    if not os.path.exists(pos_dir):
        raise FileNotFoundError(f"Positive samples directory not found: {pos_dir}")
    if not os.path.exists(neg_dir):
        raise FileNotFoundError(f"Negative samples directory not found: {neg_dir}")

    if rank == 0:
        print(f"Positive samples: {pos_dir} [OK]")
        print(f"Negative samples: {neg_dir} [OK]")
        print("-" * 40)


def save_loss_curve(loss_log, phase_boundaries, out_png: str, out_csv: str | None = None):
    """
    Save training loss curve as PNG and optional CSV.
    loss_log: List[(global_epoch, phase_name, avg_loss)]
    phase_boundaries: List[(epoch_index, phase_name)]
    """
    if not loss_log:
        print("[LossPlot] No loss data to plot.")
        return
    epochs = [e for e, _, _ in loss_log]
    losses = [l for _, _, l in loss_log]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Global Epoch")
    plt.ylabel("Avg Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.4)

    for e, name in phase_boundaries:
        plt.axvline(x=e - 0.5, linestyle="--", linewidth=1)
        plt.text(e - 0.45, max(losses), name, rotation=90, va="bottom", ha="left", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[LossPlot] Saved loss curve to: {out_png}")

    if out_csv:
        try:
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["global_epoch", "phase_name", "avg_loss"])
                for row in loss_log:
                    w.writerow(row)
            print(f"[LossPlot] Saved loss CSV to: {out_csv}")
        except Exception as e:
            print(f"[LossPlot] Failed to save CSV: {e}")


def _csv_ints(s: str):
    """Parse comma-separated integers."""
    s = s.strip()
    if not s:
        return []
    if s == "-1":
        return [-1]
    return [int(x) for x in s.split(",")]


def _csv_list(s: str):
    """Parse comma-separated strings."""
    s = s.strip()
    if not s:
        return []
    return [x for x in s.split(",")]
