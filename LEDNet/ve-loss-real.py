import matplotlib.pyplot as plt
import re
import numpy as np
import os

# ================= CẤU HÌNH =================
LOG_FILE_PATH = r"D:\DH\XLA\LEDNet\baocao-lednet\full-log.txt"
OUTPUT_IMAGE = "BieuDo_Loss_REAL_FINAL.png"
# ============================================

def parse_log_fix_comma(file_path):
    iters = []
    losses = []
    
    if not os.path.exists(file_path):
        print(f"LỖI: Không tìm thấy file {file_path}")
        return [], []

    print(f"Đang đọc file: {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if 'l_pix:' in line:
                try:
                    # 1. Tìm Loss (số thực)
                    loss_match = re.search(r"l_pix:\s*([0-9.e-]+)", line)
                    
                    # 2. Tìm Iter (QUAN TRỌNG: Chấp nhận cả dấu phẩy)
                    # Regex này tìm số có chứa dấu phẩy, ví dụ: "54,000"
                    iter_match = re.search(r"iter:\s*([\d,]+)", line, re.IGNORECASE)
                    
                    if loss_match and iter_match:
                        loss_val = float(loss_match.group(1))
                        
                        # --- XỬ LÝ DẤU PHẨY Ở ĐÂY ---
                        iter_str = iter_match.group(1).replace(',', '') # Xóa dấu phẩy: "54,000" -> "54000"
                        iter_val = int(iter_str)
                        
                        iters.append(iter_val)
                        losses.append(loss_val)
                        
                except Exception:
                    continue 

    return iters, losses

def smooth_curve(points, factor=0.98):
    """Làm mượt mạnh để đường line đẹp"""
    smoothed = []
    for point in points:
        if len(smoothed) > 0:
            prev = smoothed[-1]
            smoothed.append(prev * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

def draw_chart(iters, losses):
    if not iters:
        print("LỖI: Không đọc được dữ liệu!")
        return

    print(f"-> Đã đọc thành công {len(iters)} dòng log.")
    print(f"-> Iteration cuối cùng đọc được: {iters[-1]}") 
    # (Nếu số này hiện 54000 là Thành Công!)

    plt.figure(figsize=(12, 6))
    
    # Vẽ Raw Loss (mờ)
    plt.plot(iters, losses, color='silver', alpha=0.5, linewidth=0.5, label='Raw Loss')
    
    # Vẽ Smoothed Loss (đậm)
    losses_smooth = smooth_curve(losses)
    plt.plot(iters, losses_smooth, color='#D32F2F', linewidth=2, label='Smoothed Loss')

    plt.title(f'Training Loss Convergence (Total: {iters[-1]} steps)', fontsize=15, fontweight='bold')
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('L1 Pixel Loss', fontsize=12)
    
    # Format trục số cho đẹp (VD: 50,000)
    current_values = plt.gca().get_xticks()
    plt.gca().set_xticklabels(['{:,.0f}'.format(x) for x in current_values])
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Xong! Mở file '{OUTPUT_IMAGE}' để xem hàng Real.")
    plt.show()

if __name__ == "__main__":
    iters, losses = parse_log_fix_comma(LOG_FILE_PATH)
    draw_chart(iters, losses)