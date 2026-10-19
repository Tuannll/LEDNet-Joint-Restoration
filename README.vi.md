<div align="center">

<img src="assets/LEDNet_LOLBlur_logo.png" alt="LEDNet Architecture Logo" width="850">

# 🌟 LEDNet: Joint Low-Light Enhancement and Deblurring

**Khôi phục và nâng cao chất lượng ảnh chụp đêm bị rung mờ và thiếu sáng đồng thời dựa trên kiến trúc LEDNet (ECCV 2022) và framework BasicSR.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Tuannll/LEDNet-Joint-Restoration/blob/main/notebooks/LEDNet_Demo_Colab.ipynb)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![Framework: BasicSR](https://img.shields.io/badge/Framework-BasicSR-green.svg)](https://github.com/XPixelGroup/BasicSR)
[![UI: Gradio](https://img.shields.io/badge/Demo-Gradio-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README.md) | **Tiếng Việt**

**[📌 Giới thiệu](#problem-formulation) • [🔬 Kiến trúc](#architecture) • [📊 Benchmark](#benchmark) • [🚀 Khởi động nhanh](#quick-start) • [📁 Cấu trúc](#structure) • [📜 Trích dẫn](#citation)**

<br/>

<img src="./assets/before_after.png" alt="Before and After Comparison" width="850" />

</div>

---

> **Đồ án môn học: Xử Lý Ảnh (Image Processing - XLA)**  
> **Chủ đề:** Khôi phục và nâng cao chất lượng ảnh chụp đêm bị rung mờ và thiếu sáng đồng thời (*Joint Low-Light Enhancement and Deblurring*) dựa trên kiến trúc **LEDNet (ECCV 2022)** và framework **BasicSR**.

---

<a id="problem-formulation"></a>
## 📌 1. Giới thiệu Bài toán (Problem Formulation)

Trong điều kiện chụp đêm hoặc thiếu sáng nghiêm trọng, người chụp thường phải kéo dài thời gian phơi sáng (*long exposure*) hoặc tăng độ nhạy sáng (*high ISO*). Điều này dẫn đến hai suy biến quang học song hành: **thiếu sáng trầm trọng (low-light)** và **nhiễu rung mờ chuyển động (motion blur & noise)**.

```text
                  ┌──────────────────────────────────────────────┐
                  │           NGHỊCH LÝ XỬ LÝ TUẦN TỰ            │
                  └──────────────────────────────────────────────┘
               Tăng sáng trước               Khử mờ trước
                     │                             │
                     ▼                             ▼
       Khuếch đại nhược điểm mờ &      Thuật toán thất bại vì vùng tối
       nhiễu hạt lên gấp nhiều lần     không đủ gradient để ước lượng kernel
```

**Giải pháp của LEDNet:** Tích hợp mô hình học sâu đơn giai đoạn (*single-stage end-to-end*), ước lượng đường cong bù sáng phi tuyến kết hợp dự đoán kernel chập động học (*spatially-variant dynamic filtering*) để khôi phục ảnh sắc nét và đủ sáng tức thì.

---

<a id="architecture"></a>
## 🔬 2. Kiến trúc Mạng & Đổi mới Kỹ thuật (Architecture)

Mô hình LEDNet được xây dựng dựa trên mạng U-Net đa tỉ lệ nâng cao với 3 thành phần cốt lõi:

1. **Pyramid Pooling Module (PPM):**
   - Trích xuất đặc trưng không gian đa phạm vi bằng 4 bin pooling `(1, 2, 3, 6)`.
   - Giúp mạng nắm bắt bối cảnh ánh sáng toàn cục (*global illumination context*), hạn chế hiện tượng bù sáng không đều hoặc halo artifact.
2. **Curve Channel Attention Layer (CurveCALayer):**
   - Kế thừa cơ chế ước lượng đường cong bậc cao từ Zero-DCE:
     $$
     \mathcal{F}_n(x) = \mathcal{F}_{n-1}(x) + \mathcal{A}_n \cdot \mathcal{F}_{n-1}(x) \cdot (1 - \mathcal{F}_{n-1}(x))
     $$
   - Mở rộng dải dynamic range linh hoạt trên từng kênh đặc trưng mà không làm cháy sáng các vùng có sẵn nguồn sáng (đèn đường, biển hiệu).
3. **Dynamic Filtering (`KernelConv2D` & `conv_fac_k`):**
   - Khác với phép tích chập tĩnh với trọng số cố định, mạng tự dự đoán kernel biến thiên theo tọa độ điểm ảnh.
   - Thích ứng chính xác với các vệt mờ chuyển động phi tuyến theo hướng bất kỳ.
4. **Hàm mất mát hỗn hợp (Multi-term Loss Formulation):**
   $$
   \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pix}} (L_1) + \lambda_{\text{per}} \mathcal{L}_{\text{perceptual}} (\text{VGG-19}) + \lambda_{\text{side}} \mathcal{L}_{\text{side}}
   $$

---

<a id="benchmark"></a>
## 📊 3. Kết quả Thực nghiệm & Đánh giá Định lượng (Benchmark)

Thực nghiệm đo đạc độc lập trên tập kiểm thử phân bố phức tạp (đối chiếu giữa Ground Truth, ảnh đầu vào tối mờ, mô hình Baseline LEDNet và mô hình Fine-tuned của đồ án):

| Phương pháp | PSNR (dB) ↑ | SSIM (0-1) ↑ | VoL (Độ sắc nét) ↑ | UER (Tỷ lệ pixel tối < 20) ↓ | Latency / Throughput |
|---|:---:|:---:|:---:|:---:|:---:|
| **Ảnh đầu vào (Input)** | — | — | 18.42 | 48.75% | — |
| **Baseline LEDNet (ECCV)** | 24.18 | 0.8214 | 142.60 | 4.12% | ~38 ms (26.3 FPS) |
| **Our Fine-tuned LEDNet** | **26.85** | **0.8650** | **215.30** | **1.85%** | **~38 ms (26.3 FPS)** |

*Ghi chú:*
- **VoL (Variance of Laplacian):** Càng cao ảnh càng sắc nét, phản ánh khả năng khử rung mờ.
- **UER (Under-Exposure Ratio):** Tỷ lệ phần trăm điểm ảnh có cường độ xám $< 20$. Càng thấp ảnh càng đủ sáng.

---

<a id="quick-start"></a>
## 🚀 4. Cài đặt & Hướng dẫn Sử dụng (Quick Start)

### 4.1. Chuẩn bị môi trường
```bash
# Clone repository
git clone https://github.com/Tuannll/LEDNet-Joint-Restoration.git
cd LEDNet-Joint-Restoration

# Tạo môi trường ảo Python 3.8 / 3.9
conda create -n lednet python=3.8 -y
conda activate lednet

# Cài đặt thư viện phụ thuộc
pip install -r requirements.txt
```

### 4.2. Tải Checkpoints

> [!NOTE]
> Checkpoint fine-tuned `lednet-mytrain.pth` được lưu trữ tại GitHub Releases v1.0.0. Đảm bảo đường truyền ổn định khi tải trọng số mô hình (~100MB).

Tải weights vào thư mục `LEDNet/weights/`:
```bash
cd LEDNet
# Tải checkpoint fine-tuned chính thức của đồ án (Release v1.0.0)
python -c "from basicsr.utils.download_util import load_file_from_url; load_file_from_url('https://github.com/Tuannll/LEDNet-Joint-Restoration/releases/download/v1.0.0/lednet-mytrain.pth', model_dir='weights')"

# (Tùy chọn) Tải weights gốc ECCV 2022
python -c "from basicsr.utils.download_util import load_file_from_url; load_file_from_url('https://github.com/sczhou/LEDNet/releases/download/v0.1.0/lednet.pth', model_dir='weights')"
```

### 4.3. Chạy Inference (CLI)
```bash
# Phục hồi ảnh trong thư mục inputs/WorstPics
python inference.py --input inputs/WorstPics --output results/output --benchmark

# Chạy trên ảnh đơn lẻ tùy chọn
python inference.py --input path/to/dark_blurry_image.png --output results/single_test
```

### 4.4. Khởi chạy Giao diện Tương tác Web (Gradio Web UI)

> [!TIP]
> Thêm cờ `--share` (`python app.py --share`) để tạo đường dẫn public trực tuyến (Gradio Live link) có thời hạn 72 giờ, thuận tiện cho việc trình chiếu demo từ xa mà không cần cấu hình port forwarding.

```bash
python app.py
```
Truy cập tại: `http://127.0.0.1:7860`

---

<a id="structure"></a>
## 📁 5. Cấu trúc Thư mục Dự án

```text
LEDNet-Joint-Restoration/
├── assets/                  # Ảnh minh họa, biểu đồ loss và logo cho README
├── docs/                    # Tài liệu học thuật đồ án
├── notebooks/               # Jupyter / Colab Demo
├── LEDNet/                  # Mã nguồn chính của mô hình
│   ├── basicsr/             # Framework BasicSR tùy biến
│   │   └── archs/           # Kiến trúc LEDNet, CurveCALayer, KernelConv2D
│   ├── options/             # File cấu hình YAML fine-tune và validation
│   ├── inputs/WorstPics/    # Tập ảnh mẫu kiểm thử độ suy biến cao
│   ├── weights/             # Nơi chứa checkpoints (.pth)
│   ├── inference.py         # Unified CLI inference & latency benchmark
│   ├── evaluate.py          # Script đánh giá PSNR, SSIM, VoL, UER
│   └── app.py               # Giao diện Web tương tác Gradio
├── requirements.txt         # Danh mục thư viện phụ thuộc
└── README.md                # Tài liệu hướng dẫn đồ án
```

---

<a id="citation"></a>
## 📜 6. Trích dẫn & Lời cảm ơn (Citation)

```bibtex
@inproceedings{zhou2022lednet,
  title={LEDNet: Joint Low-Light Enhancement and Deblurring in the Dark},
  author={Zhou, Shangchen and Li, Chongyi and Loy, Chen Change},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

---

<a id="license"></a>
## 📄 License

Dự án phát hành theo giấy phép [MIT License](LICENSE).
