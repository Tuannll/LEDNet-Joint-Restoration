"""
LEDNet Training Loss Curve Plotter
Parses training log file (BasicSR format), extracts iterations and loss (l_pix),
and generates smoothed high-resolution convergence curves.
"""

import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np


def parse_log(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Log file not found: {file_path}")

    sessions = []
    current_iters = []
    current_losses = []
    last_iter = -1

    print(f"Reading log: {file_path}...")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'l_pix:' in line:
                try:
                    loss_match = re.search(r"l_pix:\s*([0-9.e-]+)", line)
                    iter_match = re.search(r"iter:\s*(\d+)", line, re.IGNORECASE)

                    if loss_match:
                        loss_val = float(loss_match.group(1))
                        if iter_match:
                            iter_val = int(iter_match.group(1))
                        else:
                            iter_val = (current_iters[-1] + 100) if current_iters else 0

                        if iter_val < last_iter:
                            if len(current_iters) > 10:
                                sessions.append((current_iters, current_losses))
                            current_iters = []
                            current_losses = []

                        current_iters.append(iter_val)
                        current_losses.append(loss_val)
                        last_iter = iter_val
                except Exception:
                    continue

    if len(current_iters) > 10:
        sessions.append((current_iters, current_losses))

    if not sessions:
        return [], []

    # Pick the longest training session
    best_session = max(sessions, key=lambda s: s[0][-1] if s[0] else 0)
    print(f"Selected longest training trajectory (up to iter {best_session[0][-1]}).")
    return best_session[0], best_session[1]


def smooth_curve(points, factor=0.99):
    smoothed = []
    for p in points:
        if smoothed:
            smoothed.append(smoothed[-1] * factor + p * (1 - factor))
        else:
            smoothed.append(p)
    return smoothed


def plot_loss(iters, losses, output_path, smooth_factor=0.995, dpi=300):
    if not iters:
        print("Error: No valid loss points found.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor('#ffffff')

    ax.plot(iters, losses, color='#CBD5E0', alpha=0.5, linewidth=0.7, label='Raw L1 Loss')
    smoothed = smooth_curve(losses, factor=smooth_factor)
    ax.plot(iters, smoothed, color='#C53030', linewidth=2.0, label=f'Exponential Moving Avg (α={smooth_factor})')

    mean_loss = float(np.mean(losses))
    ax.set_ylim(0, mean_loss * 2.8)
    ax.ticklabel_format(style='plain', axis='x')

    ax.set_title('LEDNet Fine-tuning Loss Convergence Curve', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Pixel L1 Loss ($L_{pix}$)', fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    print(f"Loss plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot LEDNet Training Loss")
    parser.add_argument('--log', '-l', type=str, default='baocao-lednet/full-log.txt',
                        help='Path to BasicSR training log file')
    parser.add_argument('--output', '-o', type=str, default='assets/training_loss.png',
                        help='Output path for generated loss curve image')
    parser.add_argument('--smooth', '-s', type=float, default=0.995,
                        help='Smoothing factor between 0.0 and 1.0 (default: 0.995)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Image resolution DPI (default: 300)')
    args = parser.parse_args()

    iters, losses = parse_log(args.log)
    plot_loss(iters, losses, args.output, smooth_factor=args.smooth, dpi=args.dpi)


if __name__ == '__main__':
    main()
