"""
LEDNet Quantitative Evaluation Script
Computes PSNR, SSIM, Variance of Laplacian (VoL - sharpness), and Under-Exposure Ratio (UER).
Outputs terminal summary table, JSON report, and benchmark comparison plots.
"""

import argparse
import json
import os
import time
import warnings
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

warnings.filterwarnings("ignore")


def calculate_vol(image):
    """Variance of Laplacian: higher means sharper image (better deblurring)."""
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_uer(image, threshold=20):
    """Under-exposure Ratio: percentage of pixels with gray value < threshold."""
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num_dark = np.sum(gray < threshold)
    return float(num_dark / gray.size)


def safe_psnr(img_gt, img_pred):
    try:
        val = psnr(img_gt, img_pred)
        if np.isinf(val) or np.isnan(val):
            return 100.0
        return float(val)
    except Exception:
        return 0.0


def evaluate_dataset(p_gt, p_in, p_base, p_fine, max_imgs=None):
    if not os.path.exists(p_gt):
        raise FileNotFoundError(f"Ground Truth directory not found: {p_gt}")

    img_names = sorted(os.listdir(p_gt))
    if max_imgs and max_imgs > 0:
        img_names = img_names[:max_imgs]

    total = len(img_names)
    print(f"Evaluating {total} images...")

    results = {
        'psnr_base': [], 'psnr_fine': [],
        'ssim_base': [], 'ssim_fine': [],
        'vol_in': [], 'vol_base': [], 'vol_fine': [],
        'uer_in': [], 'uer_base': [], 'uer_fine': []
    }

    start_t = time.time()
    valid_count = 0

    for idx, name in enumerate(img_names, 1):
        f_gt = os.path.join(p_gt, name)
        f_in = os.path.join(p_in, name)
        f_base = os.path.join(p_base, name)
        f_fine = os.path.join(p_fine, name)

        img_gt = cv2.imread(f_gt)
        img_in = cv2.imread(f_in)
        img_base = cv2.imread(f_base)
        img_fine = cv2.imread(f_fine)

        if any(x is None for x in [img_gt, img_in, img_base, img_fine]):
            continue

        valid_count += 1

        # PSNR & SSIM
        results['psnr_base'].append(safe_psnr(img_gt, img_base))
        results['psnr_fine'].append(safe_psnr(img_gt, img_fine))
        results['ssim_base'].append(float(ssim(img_gt, img_base, channel_axis=2)))
        results['ssim_fine'].append(float(ssim(img_gt, img_fine, channel_axis=2)))

        # VoL (Sharpness)
        results['vol_in'].append(calculate_vol(img_in))
        results['vol_base'].append(calculate_vol(img_base))
        results['vol_fine'].append(calculate_vol(img_fine))

        # UER (Darkness ratio)
        results['uer_in'].append(calculate_uer(img_in))
        results['uer_base'].append(calculate_uer(img_base))
        results['uer_fine'].append(calculate_uer(img_fine))

        if idx % 50 == 0 or idx == total:
            elapsed = time.time() - start_t
            print(f"  [{idx}/{total}] Processed... ({elapsed:.1f}s)")

    return results, valid_count


def generate_plots(results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#ffffff')

    # 1. PSNR Bar Chart
    mean_psnr_base = np.mean(results['psnr_base'])
    mean_psnr_fine = np.mean(results['psnr_fine'])
    bars1 = axs[0, 0].bar(['Baseline', 'Our Finetune'], [mean_psnr_base, mean_psnr_fine],
                          color=['#4A5568', '#2B6CB0'], width=0.5)
    axs[0, 0].set_title('Average PSNR (dB) ↑ (Higher is better)', fontsize=12, fontweight='bold')
    axs[0, 0].set_ylabel('PSNR (dB)')
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.5)
    axs[0, 0].bar_label(bars1, fmt='%.2f dB', padding=3, fontweight='bold')

    # 2. SSIM Bar Chart
    mean_ssim_base = np.mean(results['ssim_base'])
    mean_ssim_fine = np.mean(results['ssim_fine'])
    bars2 = axs[0, 1].bar(['Baseline', 'Our Finetune'], [mean_ssim_base, mean_ssim_fine],
                          color=['#4A5568', '#2B6CB0'], width=0.5)
    axs[0, 1].set_title('Average SSIM ↑ (Higher is better)', fontsize=12, fontweight='bold')
    axs[0, 1].set_ylabel('SSIM (0 - 1)')
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.5)
    axs[0, 1].bar_label(bars2, fmt='%.4f', padding=3, fontweight='bold')

    # 3. VoL Distribution (Sharpness)
    axs[1, 0].hist(results['vol_in'], bins=40, alpha=0.4, label='Input', color='#E53E3E')
    axs[1, 0].hist(results['vol_base'], bins=40, alpha=0.5, label='Baseline', color='#718096')
    axs[1, 0].hist(results['vol_fine'], bins=40, alpha=0.6, label='Our Finetune', color='#319795')
    axs[1, 0].set_title('Sharpness (Variance of Laplacian) Distribution ↑', fontsize=12, fontweight='bold')
    axs[1, 0].set_xlabel('VoL Value')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.4)

    # 4. UER Distribution (Darkness)
    axs[1, 1].hist(results['uer_in'], bins=40, alpha=0.4, label='Input', color='#E53E3E')
    axs[1, 1].hist(results['uer_base'], bins=40, alpha=0.5, label='Baseline', color='#718096')
    axs[1, 1].hist(results['uer_fine'], bins=40, alpha=0.6, label='Our Finetune', color='#319795')
    axs[1, 1].set_title('Under-Exposure Ratio (UER) Distribution ↓ (Lower is better)', fontsize=12, fontweight='bold')
    axs[1, 1].set_xlabel('UER Value')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Benchmark chart saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="LEDNet Quantitative Evaluation")
    parser.add_argument('--gt', type=str, default='baocao-lednet/Evaluation/gt',
                        help='Path to ground-truth directory')
    parser.add_argument('--input', type=str, default='baocao-lednet/Evaluation/input',
                        help='Path to input low-light blurry images')
    parser.add_argument('--baseline', type=str, default='baocao-lednet/Evaluation/baseline',
                        help='Path to baseline predictions')
    parser.add_argument('--finetune', type=str, default='baocao-lednet/Evaluation/finetune',
                        help='Path to finetuned predictions')
    parser.add_argument('--output_dir', type=str, default='assets',
                        help='Directory to save evaluation results and plot')
    parser.add_argument('--max_imgs', type=int, default=None,
                        help='Max number of images to evaluate (for quick test)')
    args = parser.parse_args()

    results, count = evaluate_dataset(args.gt, args.input, args.baseline, args.finetune, args.max_imgs)
    if count == 0:
        print("No valid paired images found for evaluation.")
        return

    summary = {
        'total_evaluated': count,
        'metrics': {
            'psnr': {
                'baseline_mean': float(np.mean(results['psnr_base'])),
                'baseline_std': float(np.std(results['psnr_base'])),
                'finetune_mean': float(np.mean(results['psnr_fine'])),
                'finetune_std': float(np.std(results['psnr_fine'])),
            },
            'ssim': {
                'baseline_mean': float(np.mean(results['ssim_base'])),
                'baseline_std': float(np.std(results['ssim_base'])),
                'finetune_mean': float(np.mean(results['ssim_fine'])),
                'finetune_std': float(np.std(results['ssim_fine'])),
            },
            'vol_sharpness': {
                'input_mean': float(np.mean(results['vol_in'])),
                'baseline_mean': float(np.mean(results['vol_base'])),
                'finetune_mean': float(np.mean(results['vol_fine'])),
            },
            'uer_darkness': {
                'input_mean': float(np.mean(results['uer_in'])),
                'baseline_mean': float(np.mean(results['uer_base'])),
                'finetune_mean': float(np.mean(results['uer_fine'])),
            }
        }
    }

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, 'benchmark_summary.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    plot_path = os.path.join(args.output_dir, 'evaluation_benchmark.png')
    generate_plots(results, plot_path)

    # Print terminal markdown table
    print("\n" + "=" * 65)
    print("                 BENCHMARK EVALUATION RESULTS")
    print("=" * 65)
    print(f"{'Method':<16} | {'PSNR (dB) ↑':<12} | {'SSIM ↑':<10} | {'VoL ↑':<10} | {'UER ↓':<10}")
    print("-" * 65)
    print(f"{'Input':<16} | {'-':<12} | {'-':<10} | {summary['metrics']['vol_sharpness']['input_mean']:<10.2f} | {summary['metrics']['uer_darkness']['input_mean']:<10.4f}")
    print(f"{'Baseline LEDNet':<16} | {summary['metrics']['psnr']['baseline_mean']:<12.2f} | {summary['metrics']['ssim']['baseline_mean']:<10.4f} | {summary['metrics']['vol_sharpness']['baseline_mean']:<10.2f} | {summary['metrics']['uer_darkness']['baseline_mean']:<10.4f}")
    print(f"{'Our Finetune':<16} | {summary['metrics']['psnr']['finetune_mean']:<12.2f} | {summary['metrics']['ssim']['finetune_mean']:<10.4f} | {summary['metrics']['vol_sharpness']['finetune_mean']:<10.2f} | {summary['metrics']['uer_darkness']['finetune_mean']:<10.4f}")
    print("=" * 65)
    print(f"Summary JSON: {json_path}")
    print(f"Benchmark Plot: {plot_path}\n")


if __name__ == '__main__':
    main()
