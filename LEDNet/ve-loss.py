import matplotlib.pyplot as plt
import re
import numpy as np

# ================= CẤU HÌNH =================
# Đường dẫn file log của bạn (Giữ nguyên file cũ cũng được, code tự lọc)
LOG_FILE_PATH = r"D:\DH\XLA\LEDNet\baocao-lednet\full-log.txt"
OUTPUT_IMAGE = "BieuDo_Loss_Chuan_SOTA.png"
# ============================================

def parse_log(file_path):
    sessions = [] 
    current_session_iters = []
    current_session_losses = []
    
    last_iter = -1
    
    print(f"Đang đọc file: {file_path}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
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
                            iter_val = (current_session_iters[-1] + 100) if current_session_iters else 0

                        # Nếu iter giảm đột ngột -> Là phiên train mới -> Lưu phiên cũ lại
                        if iter_val < last_iter:
                            if len(current_session_iters) > 10: # Chỉ lưu nếu session có ít nhất 10 điểm
                                sessions.append((current_session_iters, current_session_losses))
                            current_session_iters = []
                            current_session_losses = []
                        
                        current_session_iters.append(iter_val)
                        current_session_losses.append(loss_val)
                        last_iter = iter_val
                        
                except Exception:
                    continue

    # Lưu nốt session cuối cùng
    if len(current_session_iters) > 10:
        sessions.append((current_session_iters, current_session_losses))
    
    if sessions:
        print(f"Phát hiện tổng cộng {len(sessions)} đoạn train trong file log.")
        
        # --- LOGIC MỚI: TÌM ĐOẠN TRAIN DÀI NHẤT ---
        # Thay vì lấy [-1], ta dùng hàm max để tìm đoạn có nhiều iteration nhất
        best_session = max(sessions, key=lambda x: x[0][-1] if x[0] else 0)
        
        max_iter = best_session[0][-1]
        print(f"-> Đã tự động chọn đoạn train DÀI NHẤT (kết thúc ở iter {max_iter}) để vẽ.")
        return best_session
    else:
        return [], []

def smooth_curve(points, factor=0.98):
    """Làm mượt cực mạnh cho 54k điểm dữ liệu"""
    smoothed_points = []
    for point in points:
        if len(smoothed_points) > 0:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def draw_loss_chart(iters, losses):
    if not iters:
        print("LỖI: Không tìm thấy dữ liệu Loss nào!")
        return

    plt.figure(figsize=(12, 6)) # Kéo dài ra chút cho đẹp
    
    # 1. Vẽ Raw Loss (mờ hơn nữa vì dữ liệu dày đặc)
    plt.plot(iters, losses, color='silver', alpha=0.4, linewidth=0.5, label='Raw Loss')
    
    # 2. Vẽ Smoothed Loss (Đậm, đẹp)
    # Tăng factor làm mượt vì 54k điểm sẽ rất nhiễu
    losses_smooth = smooth_curve(losses, factor=0.995) 
    plt.plot(iters, losses_smooth, color='#D32F2F', linewidth=2, label='Smoothed Loss')

    # 3. Zoom hợp lý
    mean_loss = np.mean(losses)
    plt.ylim(0, mean_loss * 2.5) 
    
    # Format trục số cho đẹp (VD: 50000 thay vì 5e4)
    plt.ticklabel_format(style='plain', axis='x')

    plt.title('Training Loss Convergence (Quá trình hội tụ)', fontsize=16, fontweight='bold')
    plt.xlabel('Iterations (Vòng lặp)', fontsize=12)
    plt.ylabel('L1 Pixel Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Xong! Đã lưu biểu đồ chuẩn SOTA vào: {OUTPUT_IMAGE}")
    plt.show()

if __name__ == "__main__":
    iters, losses = parse_log(LOG_FILE_PATH)
    draw_loss_chart(iters, losses)