import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sys

# ================= CẤU HÌNH ĐƯỜNG DẪN (Sửa lại nếu cần) =================
PATH_INPUT    = r"D:\DH\XLA\LEDNet\baocao-lednet\Evaluation\input"
PATH_BASELINE = r"D:\DH\XLA\LEDNet\baocao-lednet\Evaluation\baseline"
PATH_FINETUNE = r"D:\DH\XLA\LEDNet\baocao-lednet\Evaluation\finetune"
OUTPUT_IMAGE  = "BieuDo_UER_Chuan.png"
# ========================================================================

def calculate_uer(image_path, threshold=20):
    """Tính tỷ lệ điểm ảnh tối (thiếu sáng)"""
    if not os.path.exists(image_path):
        return None
        
    # Đọc ảnh grayscale để xử lý nhanh hơn
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    # Đếm số pixel nhỏ hơn ngưỡng (tối đen)
    # img < threshold trả về ma trận True/False, sum sẽ đếm số lượng True
    num_dark_pixels = np.sum(img < threshold)
    
    # Tính tỷ lệ (0.0 -> 1.0)
    return num_dark_pixels / img.size

def collect_data():
    uer_input = []
    uer_base = []
    uer_ft = []
    
    # Kiểm tra đường dẫn input
    if not os.path.exists(PATH_INPUT):
        print(f"LỖI: Không tìm thấy thư mục Input tại: {PATH_INPUT}")
        return [], [], []

    img_list = os.listdir(PATH_INPUT)
    total_imgs = len(img_list)
    print(f"-> Tìm thấy tổng cộng {total_imgs} ảnh. Bắt đầu tính toán UER...")

    # --- VÒNG LẶP CÓ HIỂN THỊ % ---
    for i, name in enumerate(img_list):
        # Đường dẫn file
        p_in   = os.path.join(PATH_INPUT, name)
        p_base = os.path.join(PATH_BASELINE, name)
        p_ft   = os.path.join(PATH_FINETUNE, name)
        
        # Tính toán
        val_in   = calculate_uer(p_in)
        val_base = calculate_uer(p_base)
        val_ft   = calculate_uer(p_ft)
        
        # Chỉ thêm vào list nếu đọc được cả 3 ảnh
        if val_in is not None and val_base is not None and val_ft is not None:
            uer_input.append(val_in)
            uer_base.append(val_base)
            uer_ft.append(val_ft)

        # --- LOGIC HIỂN THỊ TIẾN ĐỘ ---
        # Cứ mỗi 10 ảnh hoặc ảnh cuối cùng thì in ra màn hình 1 lần
        if (i + 1) % 20 == 0 or (i + 1) == total_imgs:
            percent = (i + 1) / total_imgs * 100
            # Dùng sys.stdout để in đè lên dòng cũ cho gọn
            sys.stdout.write(f"\r   ... Đang chạy: [{percent:6.2f}%] - Ảnh thứ {i + 1}/{total_imgs}")
            sys.stdout.flush()

    print("\n-> Hoàn thành tính toán! Đang vẽ biểu đồ...")
    return uer_input, uer_base, uer_ft

def draw_uer_chart(uer_input, uer_base, uer_ft):
    if not uer_input:
        print("LỖI: Không có dữ liệu để vẽ! (Kiểm tra lại đường dẫn folder)")
        return

    plt.figure(figsize=(10, 6))
    
    # --- CẤU HÌNH BIN (Theo yêu cầu của bạn) ---
    # Chia khoảng từ 0 đến 1 thành 30 cột đều nhau
    bins = np.linspace(0, 1, 30)

    # --- VẼ HISTOGRAM ---
    # 1. Input (Màu đỏ)
    plt.hist(uer_input, bins=bins, color='red', alpha=0.4, 
             density=True, label="Input (Ảnh tối)", edgecolor='white', linewidth=0.5)
    
    # 2. Baseline (Màu xám)
    plt.hist(uer_base, bins=bins, color='gray', alpha=0.4, 
             density=True, label="Baseline (Tác giả)", edgecolor='white', linewidth=0.5)
    
    # 3. Finetune (Màu xanh)
    plt.hist(uer_ft, bins=bins, color='green', alpha=0.5, 
             density=True, label="Finetune (Của tôi)", edgecolor='white', linewidth=0.5)

    # Trang trí
    plt.title("Under-Exposed Ratio (UER) Distribution", fontsize=14, fontweight='bold')
    plt.xlabel("UER Ratio (0=Sáng, 1=Tối)", fontsize=12)
    plt.ylabel("Density (Mật độ phân phối)", fontsize=12)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 1) # Cố định trục X từ 0 đến 1

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Xong! Biểu đồ UER đã lưu vào: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    in_data, base_data, ft_data = collect_data()
    draw_uer_chart(in_data, base_data, ft_data)