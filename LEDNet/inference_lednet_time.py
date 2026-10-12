# Modified by Shangchen Zhou from: https://github.com/TencentARC/GFPGAN/blob/master/inference_gfpgan.py

import os
import cv2
import argparse
import time
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from basicsr.utils import imwrite, img2tensor, tensor2img, scandir
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.registry import ARCH_REGISTRY

pretrain_model_url = {
    'lednet': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet.pth',
    'lednet_retrain': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet_retrain_500000.pth',
    'lednetgan': 'https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednetgan.pth',
}

def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default=r'd:\DH\XLA\LEDNet\inputs\WorstPics')
    parser.add_argument('--result_path', type=str, default=r'd:\DH\XLA\LEDNet\results')
    parser.add_argument('--model', type=str, default='lednet',
                        help='options: lednet, lednet_retrain, lednetgan')
    args = parser.parse_args()

    # ------------------------ input & output ------------------------
    if args.test_path.endswith('/'):
        args.test_path = args.test_path[:-1]
    if args.result_path.endswith('/'):
        args.result_path = args.result_path[:-1]

    result_root = f'{args.result_path}/{os.path.basename(args.test_path)}'
    os.makedirs(result_root, exist_ok=True)

    # ------------------ set up LEDNet network -------------------
    down_factor = 8
    net = ARCH_REGISTRY.get('LEDNet')(
        channels=[32, 64, 128, 128],
        connection=False
    ).to(device)

    assert args.model in ['lednet', 'lednet_retrain', 'lednetgan']
    ckpt_path = load_file_from_url(
        url=pretrain_model_url[args.model],
        model_dir='./weights',
        progress=True,
        file_name=None
    )

    checkpoint = torch.load(ckpt_path, map_location=device)['params']
    net.load_state_dict(checkpoint)
    net.eval()

    # -------------------- start processing ---------------------
    img_paths = sorted(list(scandir(
        args.test_path,
        suffix=('jpg', 'png'),
        recursive=True,
        full_path=True
    )))

    total_time = 0.0
    num_images = len(img_paths)

    for img_path in img_paths:
        img_name = img_path.replace(args.test_path + '/', '')
        print(f'Processing: {img_name}')

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_t = img2tensor(img / 255., bgr2rgb=True, float32=True)

        # LEDNet gốc không normalize [-1, 1]
        if args.model != 'lednet':
            normalize(img_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)

        img_t = img_t.unsqueeze(0).to(device)

        # ---------------- inference + timing ----------------
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        with torch.no_grad():
            H, W = img_t.shape[2:]
            img_t = check_image_size(img_t, down_factor)
            output_t = net(img_t)
            output_t = output_t[:, :, :H, :W]

            if args.model == 'lednet':
                output = tensor2img(output_t, rgb2bgr=True, min_max=(0, 1))
            else:
                output = tensor2img(output_t, rgb2bgr=True, min_max=(-1, 1))

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        infer_time = end_time - start_time
        total_time += infer_time

        print(f'Inference time: {infer_time:.4f} s')

        # ---------------- save result ----------------
        output = output.astype('uint8')
        save_restore_path = img_path.replace(args.test_path, result_root)
        os.makedirs(os.path.dirname(save_restore_path), exist_ok=True)
        imwrite(output, save_restore_path)

        del output_t
        torch.cuda.empty_cache()

    avg_time = total_time / num_images if num_images > 0 else 0

    print('\n==================== SUMMARY ====================')
    print(f'Device: {device}')
    print(f'Number of images: {num_images}')
    print(f'Total inference time: {total_time:.4f} s')
    print(f'Average inference time per image: {avg_time:.4f} s')
    print(f'All results are saved in: {result_root}')
