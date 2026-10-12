import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import warnings
import time

# Tắt cảnh báo chia cho 0
warnings.filterwarnings("ignore")

# ================= CẤU HÌNH ĐƯỜNG DẪN =================
# Giữ nguyên đường dẫn như bạn đã cung cấp
PATH_GT       = r"D:\DH\XLA\LEDNet\baocao-lednet\Evaluation\gt"
PATH_BASELINE = r"D:\DH\XLA\LEDNet\baocao-lednet\Evaluation\baseline"
PATH_FINETUNE = r"D:\DH\XLA\LEDNet\baocao-lednet\Evaluation\finetune"
PATH_INPUT    = r"D:\DH\XLA\LEDNet\baocao-lednet\Evaluation\input"
# ======================================================

def calculate_vol(image):
    if image is None: return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_uer(image, threshold=20):
    if image is None: return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num_dark_pixels = np.sum(gray < threshold)
    return num_dark_pixels / gray.size

def safe_psnr(img1, img2):
    try:
        val = psnr(img1, img2)
        if np.isinf(val) or np.isnan(val):
            return 100.0 
        return val
    except:
        return 0.0

def process_evaluation():
    scores_psnr = {'Baseline': [], 'Finetune': []}
    scores_ssim = {'Baseline': [], 'Finetune': []}
    vals_vol = {'Input': [], 'Baseline': [], 'Finetune': []}
    vals_uer = {'Input': [], 'Baseline': [], 'Finetune': []}

    # Kiểm tra đường dẫn tồn tại không
    if not os.path.exists(PATH_GT):
        print(f"LỖI: Không tìm thấy folder tại {PATH_GT}")
        return None, None, None, None

    img_names = os.listdir(PATH_GT)
    total_imgs = len(img_names)
    print(f"-> Tìm thấy {total_imgs} ảnh. Bắt đầu xử lý...")
    
    start_time = time.time() # Bấm giờ

    for i, name in enumerate(img_names):
        # 1. In tiến độ ra màn hình (Cứ 50 ảnh in 1 lần)
        if i % 50 == 0:
            elapsed = time.time() - start_time
            print(f"   ... Đang xử lý ảnh thứ {i}/{total_imgs} ({name}) - Đã chạy {elapsed:.1f}s")

        p_gt = os.path.join(PATH_GT, name)
        p_base = os.path.join(PATH_BASELINE, name)
        p_fine = os.path.join(PATH_FINETUNE, name)
        p_in = os.path.join(PATH_INPUT, name)

        img_gt = cv2.imread(p_gt)
        img_base = cv2.imread(p_base)
        img_fine = cv2.imread(p_fine)
        img_in = cv2.imread(p_in)

        # Nếu ảnh lỗi hoặc không tìm thấy, bỏ qua
        if any(x is None for x in [img_gt, img_base, img_fine, img_in]):
            # print(f"Cảnh báo: Không đọc được bộ ảnh {name}, bỏ qua.")
            continue

        # 2. Tính toán (Phần này nặng nhất)
        scores_psnr['Baseline'].append(safe_psnr(img_gt, img_base))
        # SSIM tính rất lâu, kiên nhẫn nhé
        scores_ssim['Baseline'].append(ssim(img_gt, img_base, channel_axis=2))

        scores_psnr['Finetune'].append(safe_psnr(img_gt, img_fine))
        scores_ssim['Finetune'].append(ssim(img_gt, img_fine, channel_axis=2))

        vals_vol['Input'].append(calculate_vol(img_in))
        vals_vol['Baseline'].append(calculate_vol(img_base))
        vals_vol['Finetune'].append(calculate_vol(img_fine))

        vals_uer['Input'].append(calculate_uer(img_in))
        vals_uer['Baseline'].append(calculate_uer(img_base))
        vals_uer['Finetune'].append(calculate_uer(img_fine))

    print(f"HOÀN THÀNH! Tổng thời gian: {time.time() - start_time:.1f} giây.")
    print("Đang vẽ biểu đồ...")
    return scores_psnr, scores_ssim, vals_vol, vals_uer

def draw_charts(scores_psnr, scores_ssim, vals_vol, vals_uer):
    if scores_psnr is None: return

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # PSNR
    avg_psnr = [np.mean(scores_psnr['Baseline']), np.mean(scores_psnr['Finetune'])]
    bars = axs[0, 0].bar(['Baseline', 'Finetune'], avg_psnr, color=['gray', 'green'])
    axs[0, 0].set_title('Average PSNR (dB)')
    axs[0, 0].bar_label(bars, fmt='%.2f')

    # SSIM
    avg_ssim = [np.mean(scores_ssim['Baseline']), np.mean(scores_ssim['Finetune'])]
    bars2 = axs[0, 1].bar(['Baseline', 'Finetune'], avg_ssim, color=['gray', 'green'])
    axs[0, 1].set_title('Average SSIM (0-1)')
    axs[0, 1].bar_label(bars2, fmt='%.4f')

    # VOL
    axs[1, 0].hist(vals_vol['Input'], bins=50, alpha=0.5, label='Input', color='red')
    axs[1, 0].hist(vals_vol['Baseline'], bins=50, alpha=0.5, label='Baseline', color='gray')
    axs[1, 0].hist(vals_vol['Finetune'], bins=50, alpha=0.5, label='Finetune', color='green')
    axs[1, 0].set_title('Histogram Độ Nét (VOL)')
    axs[1, 0].legend()

    # UER
    axs[1, 1].hist(vals_uer['Input'], bins=50, alpha=0.5, label='Input', color='red')
    axs[1, 1].hist(vals_uer['Baseline'], bins=50, alpha=0.5, label='Baseline', color='gray')
    axs[1, 1].hist(vals_uer['Finetune'], bins=50, alpha=0.5, label='Finetune', color='green')
    axs[1, 1].set_title('Histogram Vùng Tối (UER)')
    axs[1, 1].legend()

    plt.tight_layout()
    plt.savefig('KetQua_DanhGia.png')
    print("Đã lưu biểu đồ vào file 'KetQua_DanhGia.png'")
    plt.show()

if __name__ == "__main__":
    p, s, v, u = process_evaluation()
    draw_charts(p, s, v, u)