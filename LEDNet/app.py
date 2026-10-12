"""
LEDNet Interactive Web Demo (Gradio)
Joint Low-Light Enhancement and Deblurring
Provides real-time enhancement, side-by-side comparison, and IQA metrics (VoL, UER, Latency).
"""

import argparse
import os
import sys
import time
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

# Ensure current directory is on python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from basicsr.utils import img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY


PRETRAIN_URLS = {
    'baseline': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet.pth',
    'retrain': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet_retrain_500000.pth',
}

CACHED_MODELS = {}


def safe_torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def get_model(model_choice='finetune', custom_ckpt=None, device='cuda'):
    key = f"{model_choice}_{device}"
    if key in CACHED_MODELS:
        return CACHED_MODELS[key]

    net = ARCH_REGISTRY.get('LEDNet')(channels=[32, 64, 128, 128], connection=False).to(device)
    use_norm = True

    if model_choice == 'finetune':
        local_path = custom_ckpt or os.path.join(current_dir, 'weights', 'lednet-mytrain.pth')
        if not os.path.exists(local_path):
            local_path = os.path.join(current_dir, 'weights', 'lednet_retrain_500000.pth')
        if os.path.exists(local_path):
            ckpt = safe_torch_load(local_path, device)
            use_norm = True
        else:
            # Fallback download
            dl_path = load_file_from_url(PRETRAIN_URLS['retrain'], model_dir=os.path.join(current_dir, 'weights'))
            ckpt = safe_torch_load(dl_path, device)
            use_norm = True
    else:  # baseline
        local_path = os.path.join(current_dir, 'weights', 'lednet.pth')
        if not os.path.exists(local_path):
            local_path = load_file_from_url(PRETRAIN_URLS['baseline'], model_dir=os.path.join(current_dir, 'weights'))
        ckpt = safe_torch_load(local_path, device)
        use_norm = False

    state_dict = ckpt['params'] if 'params' in ckpt else ckpt
    net.load_state_dict(state_dict)
    net.eval()
    CACHED_MODELS[key] = (net, use_norm)
    return net, use_norm


def pad_to_multiple(x, factor=8):
    _, _, h, w = x.size()
    pad_h = (factor - h % factor) % factor
    pad_w = (factor - w % factor) % factor
    return F.pad(x, (0, pad_w, 0, pad_h), 'reflect')


def calc_metrics(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    vol = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    uer = float(np.sum(gray < 20) / gray.size) * 100.0  # percentage
    return vol, uer


def enhance_image(input_img, model_type='finetune'):
    if input_img is None:
        return None, "Please upload or select an input image."

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net, use_norm = get_model(model_type, device=device)

    # Input image from Gradio is RGB numpy uint8
    img_bgr = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
    vol_in, uer_in = calc_metrics(img_bgr)

    # Prepare tensor
    img_t = img2tensor(img_bgr / 255., bgr2rgb=True, float32=True)
    if use_norm:
        normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    img_t = img_t.unsqueeze(0).to(device)

    # Run inference
    t0 = time.perf_counter()
    with torch.no_grad():
        h, w = img_t.shape[2:]
        img_padded = pad_to_multiple(img_t, 8)
        out_t = net(img_padded)[:, :, :h, :w]

        if not use_norm:
            out_bgr = tensor2img(out_t, rgb2bgr=True, min_max=(0, 1))
        else:
            out_bgr = tensor2img(out_t, rgb2bgr=True, min_max=(-1, 1))
    latency = (time.perf_counter() - t0) * 1000.0

    vol_out, uer_out = calc_metrics(out_bgr)
    out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)

    vol_diff = ((vol_out - vol_in) / max(vol_in, 1e-5)) * 100.0
    uer_diff = uer_out - uer_in

    stats_md = f"""
### 📊 Image Quality Assessment (IQA) & Performance
| Metric | Input Image | Restored Output | Delta Change |
|---|---|---|---|
| **Under-Exposure Ratio (UER)** | `{uer_in:.2f}%` | `{uer_out:.2f}%` | **{uer_diff:+.2f}%** (Lower is better) |
| **Sharpness (VoL)** | `{vol_in:.1f}` | `{vol_out:.1f}` | **{vol_diff:+.1f}%** (Higher is better) |
| **Latency / Throughput** | - | `{latency:.1f} ms` | **~{1000.0/max(latency, 1e-3):.1f} FPS** ({device.upper()}) |
    """
    return out_rgb, stats_md


def build_app(share=False, server_port=7860):
    example_dir = os.path.join(current_dir, 'inputs', 'WorstPics')
    examples = []
    if os.path.exists(example_dir):
        sample_names = ['C0310_0042.png', 'C0314_0092.png', '075_blur_3.png', '100_blur_8.png', '206_blur_21.png']
        for s in sample_names:
            sp = os.path.join(example_dir, s)
            if os.path.exists(sp):
                examples.append(sp)

    custom_theme = gr.themes.Soft(
        primary_hue="blue",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=custom_theme, title="LEDNet Image Restoration") as demo:
        gr.Markdown(
            """
            # 🌟 LEDNet: Joint Low-light Enhancement and Deblurring
            ### Deep Learning Final Project — Computer Vision / Image Processing (XLA)
            Upload a dark, blurry image or click a sample below to restore vivid exposure and crisp details in real-time.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input (Low-light & Motion Blur)", type="numpy")
                model_selector = gr.Radio(
                    choices=[('Our Fine-tuned Model (High Quality)', 'finetune'), ('Baseline LEDNet (Original)', 'baseline')],
                    value='finetune',
                    label="Model Checkpoint"
                )
                run_btn = gr.Button("✨ Enhance & Deblur", variant="primary", size="lg")

                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=input_image,
                        label="Sample Test Images (Challenging Cases)"
                    )

            with gr.Column(scale=1):
                output_image = gr.Image(label="Restored Output", type="numpy")
                metrics_box = gr.Markdown("### 📊 Metrics will appear after inference.")

        run_btn.click(
            fn=enhance_image,
            inputs=[input_image, model_selector],
            outputs=[output_image, metrics_box]
        )

        gr.Markdown(
            """
            ---
            **Architecture highlights:**
            - **Pyramid Pooling Module (PPM)** for multi-scale spatial context.
            - **CurveCALayer (Curve Channel Attention)** for non-linear dynamic range correction.
            - **Dynamic Filtering (`KernelConv2D`)** for adaptive local deblurring.
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="LEDNet Gradio Web App")
    parser.add_argument('--share', action='store_true', help='Create a public shareable gradio.live link')
    parser.add_argument('--port', type=int, default=7860, help='Port to serve the web demo')
    args = parser.parse_args()

    app = build_app()
    app.launch(share=args.share, server_port=args.port)


if __name__ == '__main__':
    main()
