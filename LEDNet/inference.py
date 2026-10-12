"""
LEDNet Inference Script
Unified CLI tool for Joint Low-light Enhancement and Deblurring.
Supports single image or folder input, custom or official checkpoints, and latency benchmark.
"""

import argparse
import os
import time
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from basicsr.utils import imwrite, img2tensor, tensor2img, scandir
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY

PRETRAIN_MODEL_URL = {
    'lednet': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet.pth',
    'lednet_retrain': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet_retrain_500000.pth',
    'lednetgan': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednetgan.pth',
}


def check_image_size(x, down_factor=8):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    return F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        # Fallback for PyTorch versions prior to weights_only support
        return torch.load(path, map_location=device)


def load_model(checkpoint_path=None, model_name='lednet_retrain', device='cuda'):
    net = ARCH_REGISTRY.get('LEDNet')(channels=[32, 64, 128, 128], connection=False).to(device)

    use_norm = True
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from local path: {checkpoint_path}")
        ckpt = safe_torch_load(checkpoint_path, device)
        use_norm = True  # custom fine-tune uses mean [0.5, 0.5, 0.5]
    else:
        # Check if model already in weights/
        local_weight = os.path.join('weights', f'{model_name}.pth')
        if os.path.exists(local_weight):
            ckpt_path = local_weight
            print(f"Loading pre-trained model: {ckpt_path}")
        else:
            print(f"Downloading pre-trained model '{model_name}'...")
            ckpt_path = load_file_from_url(
                url=PRETRAIN_MODEL_URL[model_name],
                model_dir='./weights',
                progress=True,
                file_name=None
            )
        ckpt = safe_torch_load(ckpt_path, device)
        use_norm = (model_name != 'lednet')

    state_dict = ckpt['params'] if 'params' in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.eval()
    return net, use_norm


def process_image(net, img_path, device, use_norm, down_factor=8):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, 0.0

    img_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
    if use_norm:
        normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    img_t = img_t.unsqueeze(0).to(device)

    start_t = time.perf_counter()
    with torch.no_grad():
        h, w = img_t.shape[2:]
        img_padded = check_image_size(img_t, down_factor)
        out_t = net(img_padded)[:, :, :h, :w]

        if not use_norm:
            out_img = tensor2img(out_t, rgb2bgr=True, min_max=(0, 1))
        else:
            out_img = tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1))
    latency = time.perf_counter() - start_t

    return out_img.astype('uint8'), latency


def main():
    parser = argparse.ArgumentParser(description="LEDNet Image Restoration Inference")
    parser.add_argument('--input', '-i', type=str, default='inputs/WorstPics',
                        help='Input image path or directory containing images')
    parser.add_argument('--output', '-o', type=str, default='results',
                        help='Directory to save restored images')
    parser.add_argument('--checkpoint', '-c', type=str, default=None,
                        help='Path to custom checkpoint (.pth)')
    parser.add_argument('--model', '-m', type=str, default='lednet_retrain',
                        choices=['lednet', 'lednet_retrain', 'lednetgan'],
                        help='Pretrained model variant when checkpoint is not specified')
    parser.add_argument('--benchmark', action='store_true',
                        help='Enable latency benchmark and print FPS statistics')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to run on (cuda or cpu). Auto-detects if omitted.')
    args = parser.parse_args()

    # Determine default checkpoint if available
    default_finetuned = os.path.join('weights', 'lednet-mytrain.pth')
    if args.checkpoint is None and os.path.exists(default_finetuned):
        args.checkpoint = default_finetuned
        print(f"Using default fine-tuned checkpoint: {default_finetuned}")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Execution device: {device}")

    net, use_norm = load_model(args.checkpoint, args.model, device)

    # Collect input files
    if os.path.isfile(args.input):
        img_paths = [args.input]
        input_dir = os.path.dirname(args.input)
    elif os.path.isdir(args.input):
        input_dir = args.input.rstrip('/\\')
        img_paths = sorted(list(scandir(args.input, suffix=('jpg', 'jpeg', 'png', 'bmp'), recursive=True, full_path=True)))
    else:
        raise FileNotFoundError(f"Input path not found: {args.input}")

    if not img_paths:
        print(f"No valid images found in {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)
    latencies = []
    print(f"Processing {len(img_paths)} image(s)...")

    for idx, path in enumerate(img_paths, 1):
        rel_path = os.path.relpath(path, input_dir) if input_dir else os.path.basename(path)
        out_path = os.path.join(args.output, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        out_img, latency = process_image(net, path, device, use_norm)
        if out_img is not None:
            imwrite(out_img, out_path)
            latencies.append(latency)
            if not args.benchmark:
                print(f"[{idx}/{len(img_paths)}] Saved: {out_path} ({latency * 1000:.1f}ms)")
            else:
                print(f"[{idx}/{len(img_paths)}] Processed: {rel_path} ({latency * 1000:.1f}ms)")

    print(f"\nCompleted! Results saved to: {args.output}")

    if args.benchmark and latencies:
        # Exclude warm-up first frame if multiple frames
        valid_latencies = latencies[1:] if len(latencies) > 1 else latencies
        avg_lat = sum(valid_latencies) / len(valid_latencies)
        fps = 1.0 / avg_lat if avg_lat > 0 else 0
        print("=" * 45)
        print("          BENCHMARK SUMMARY")
        print("=" * 45)
        print(f"Total Images:     {len(latencies)}")
        print(f"Average Latency:  {avg_lat * 1000:.2f} ms")
        print(f"Throughput:       {fps:.2f} FPS")
        print("=" * 45)


if __name__ == '__main__':
    main()
