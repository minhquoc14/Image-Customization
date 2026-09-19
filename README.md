# Customized Image Generation - Style Transfer via Stable Diffusion + LoRA Fine-Tuning

## Tổng Quan Dự Án

### Giới Thiệu

Dự án **Customized Image Generation** nghiên cứu và triển khai phương pháp chuyển đổi phong cách nghệ thuật cho ảnh sử dụng Stable Diffusion kết hợp với kỹ thuật LoRA (Low-Rank Adaptation) fine-tuning. Hệ thống cho phép người dùng cung cấp ảnh nội dung (content image) và chọn phong cách nghệ thuật (style class) để tự động tạo ra ảnh mới giữ nguyên bố cục nhưng mang đặc trưng phong cách đã chọn.

### Thành Viên Nhóm

1. **Nguyễn Khang Hy** (2352662)
2. **Phan Đức Thành Phát** (23521149)
3. **Nguyễn Minh Quốc** (23521304)

---

## Động Lực và Mục Tiêu

### Vấn Đề Hiện Tại

- Các hệ thống AI sinh ảnh hiện tại (DALL-E, Midjourney) yêu cầu prompt văn bản chính xác, khó kiểm soát kết quả
- Style transfer truyền thống (AdaIN, các phương pháp CNN-based) có giới hạn về chất lượng và độ tự nhiên
- Fine-tuning toàn bộ Stable Diffusion tốn tài nguyên và thời gian

### Giải Pháp Đề Xuất

- Sử dụng Stable Diffusion v1.5 làm base model
- Fine-tuning bằng **LoRA** (Low-Rank Adaptation) chỉ trên UNet attention layers
- Fine-tuning bằng **DreamBooth** với prior preservation
- Fine-tuning bằng **Textual Inversion** cho embedding tokens
- So sánh 3 phương pháp fine-tuning trên cùng dataset và metrics
- Nhẹ, nhanh, dễ mở rộng phong cách mới

### Mục Tiêu

- Fine-tune thành công 5 phong cách nghệ thuật (LoRA), 2 phong cách (DreamBooth), 1 phong cách (Textual Inversion)
- Ảnh sinh ra giữ bố cục content (SSIM > baseline) và thể hiện style (LPIPS vừa phải)
- Demo chạy ổn định, thời gian inference < 5s/ảnh
- Model gọn < 1 tỉ tham số, training < vài ngày

---

## Bài Toán

### Phát Biểu Bài Toán

Fine-tune mô hình Stable Diffusion để sinh ảnh theo phong cách cụ thể (style class) dựa trên ảnh gốc (content image). Mô hình học phân phối có điều kiện p(x | style), cho phép tạo ra ảnh mới giữ bố cục content nhưng mang đặc trưng của style.

### Input/Output

**Input:**

- Content_Image: Ảnh gốc giữ bố cục và nội dung chính
- Style_Class hoặc Style_Image: Lựa chọn phong cách từ thư viện có sẵn hoặc upload ảnh phong cách
- Tùy chọn: style_strength, mask vùng áp style

**Output:**

- Ảnh mới giữ bố cục content và mang phong cách tương ứng

### Ràng Buộc

1. **Content Preservation**: Giữ cấu trúc và bố cục của ảnh gốc
2. **Style Transfer**: Tái tạo texture, màu sắc, họa tiết của ảnh style
3. **Efficiency**: Model gọn, training nhanh, inference nhanh

---

## Kiến Trúc Mô Hình

### Stable Diffusion v1.5

- **Base Model**: `runwayml/stable-diffusion-v1-5`
- **Components**:
  - VAE Encoder/Decoder: Encode/decode giữa pixel space và latent space
  - UNet: Denoising network trong latent space

### LoRA (Low-Rank Adaptation)

**Ý tưởng**: Thay vì fine-tune toàn bộ UNet, chỉ thêm các low-rank matrices vào attention layers

**Công thức**: `W' = W + α·A·B` với A ∈ R^(d×r), B ∈ R^(r×d), r << d

**Ưu điểm**:

- Giảm số tham số train từ ~860M xuống ~4-8M
- Training nhanh hơn 10-20 lần
- Dễ quản lý nhiều style (mỗi style 1 checkpoint)
- Có thể kết hợp nhiều LoRA

**Fine-tune Target**: UNet attention layers (cross-attention và self-attention)

### Tại Sao Kết Hợp SD Với LoRA?

**Vấn đề của Full Fine-tuning**:

- Stable Diffusion v1.5 có ~860M parameters
- Fine-tune toàn bộ tốn nhiều tài nguyên:
  - GPU memory: ~24GB (cần GPU lớn như A100)
  - Training time: Vài ngày cho 1 style
  - Checkpoint size: ~3-4GB mỗi style
  - Khó quản lý nhiều styles (5 styles = 15-20GB)

**Giải pháp LoRA**:

- Chỉ train ~4-8M parameters (giảm 99% so với full fine-tuning)
- Training nhanh: < 6 giờ thay vì vài ngày (thực tế: ~5-6 giờ cho 1 style)
- Checkpoint nhỏ: ~4-8MB mỗi style (thay vì 3-4GB)
- Tiết kiệm GPU memory: Có thể train trên GPU nhỏ hơn (T4, P100)
- Dễ quản lý: Mỗi style 1 file LoRA nhỏ, dễ switch giữa các styles

**So sánh**:

| Phương pháp | Parameters | Checkpoint Size | Training Time | GPU Memory |
|-------------|-----------|----------------|---------------|------------|
| **Full Fine-tune** | 860M | ~3-4GB | Vài ngày | ~24GB |
| **LoRA (r=4)** | ~4-8M | ~4-8mb | < 6 giờ | ~12GB |
| **DreamBooth (attention-only)** | ~260M (30% UNet) | ~2-4GB | ~12 giờ | ~5-6GB |

**Kết luận**:

- SD: Model mạnh, đã được train sẵn, có khả năng generate ảnh tốt
- LoRA: Cách hiệu quả nhất để adapt SD cho style cụ thể - training nhanh nhất (< 6h) với ít parameters nhất (~4-8M)
- DreamBooth: Training chậm hơn (~12h) dù chỉ train 30% parameters do phải load toàn bộ model và xử lý prior preservation
- Kết hợp: Tận dụng sức mạnh của SD + training nhanh/gọn của LoRA

---

## Pipeline Chi Tiết

### 1. Chuẩn Bị Dữ Liệu

**Content Dataset**: COCO 2017

- 118k train images, 5k val images
- Ảnh thực tế đời thường, bố cục tự nhiên
- Resize về 512x512

**Style Dataset**: WikiArt

- 5 phong cách nghệ thuật được chọn
- Số lượng ảnh/phong cách: Contemporary_Realism (481), New_Realism (314), Synthetic_Cubism (216), Analytical_Cubism (110), Action_painting (98)
- LoRA training: 100 ảnh/phong cách
- DreamBooth training: 40 instance images + 200 class images/phong cách
- Textual Inversion training: 20 ảnh từ COCO dataset

### 2a. Fine-tune LoRA

**Cấu hình**:

- Base model: `runwayml/stable-diffusion-v1-5`
- Fine-tune target: UNet attention layers
- Rank: 4
- Learning rate: 1e-4
- Batch size: 2
- Steps: 1,500/phong cách
- Optimizer: AdamW
- Scheduler: Cosine

**Loss Function**:
```
L_total = α·L2 + β·LPIPS + γ·StyleLoss
```

- L2 loss: Tái tạo chi tiết ảnh
- LPIPS: Duy trì độ tự nhiên theo cảm nhận người nhìn
- Style loss (Gram matrix): Giữ họa tiết, màu sắc của style

### 2b. Fine-tune DreamBooth

**Mục tiêu**: Fine-tune UNet với prior preservation để học phong cách nghệ thuật cụ thể. Do hạn chế về phần cứng (GPU memory trên Kaggle), chúng em chỉ fine-tune **attention layers** của UNet thay vì toàn bộ UNet.

**Lý do chỉ train attention layers**:

- **Hạn chế phần cứng**: Kaggle GPU (T4/P100) có ~16GB VRAM, không đủ để train full UNet (~860M parameters) với batch size hợp lý
- **Memory requirements**: 
  - Full UNet training: ~15-16GB VRAM (model + optimizer state + activations)
  - Attention layers only: ~5-6GB VRAM (chỉ ~30% parameters cần train)
- **Trade-off**: Giảm memory usage đáng kể nhưng vẫn giữ được khả năng học style transfer hiệu quả vì attention layers là phần quan trọng nhất trong UNet để học các đặc trưng style

**Cấu hình**:

- Base model: Stable Diffusion v1.5
- **Fine-tune target: Chỉ attention layers của UNet** (cross-attention và self-attention)
- Parameters train: ~30% của UNet (~260M parameters thay vì 860M)
- Input size: 256 (giảm từ 512 để tiết kiệm memory)
- Instance images: 40 ảnh/phong cách
- Class images: 200 ảnh/phong cách (prior preservation)
- Learning rate: 1e-5
- Batch size: 1 (với gradient accumulation 16)
- Steps: 2000 per style
- Optimizer: AdamW
- Loss: MSE loss + Prior preservation loss (weight=0.6)

**Memory optimizations** (bắt buộc do hạn chế phần cứng):

- **CPU offloading**: VAE và Text Encoder ở CPU, chỉ move lên GPU khi encode
- **VAE slicing và tiling**: Chia VAE encoding thành các slice/tile nhỏ hơn
- **Attention slicing**: Chia attention mechanism thành các slice
- **Gradient checkpointing**: Trade computation for memory
- **Resolution reduction**: 512 → 256 để giảm memory cho activations
- **Gradient accumulation**: Batch size 1 với accumulation 16 để mô phỏng batch lớn hơn

**Kết quả và Hạn chế**:

- Checkpoint: Chỉ lưu attention layers đã train (~260M parameters), có thể load vào base model
- Memory usage: ~5-6GB VRAM (thay vì ~15GB nếu train full UNet)
- Chất lượng: Style transfer hoạt động nhưng chưa mạnh như full UNet training

**Tại sao kết quả chưa tối ưu?**:

1. **Chỉ train attention layers (~30% parameters)**:
   - Attention layers: Điều khiển "what to attend to" (nội dung, style concept)
   - ResNet blocks: Điều khiển "how to process" (texture, brushstrokes, rendering details)
   - **Hệ quả**: Model học được style concept nhưng thiếu texture/brushstroke details
   
2. **Hạn chế phần cứng**:
   - Kaggle GPU: T4/P100 với 16GB VRAM
   - Full UNet training cần ~20-24GB VRAM (model + optimizer state + activations)
   - Không thể train full UNet → phải chấp nhận trade-off
   
3. **Hạn chế thời gian**:
   - Kaggle timeout: 12 giờ/session
   - Training 1 style: ~12 giờ (với attention-only, ~30% parameters)
   - **Lý do chậm hơn LoRA**: Dù chỉ train 30% parameters, DreamBooth vẫn phải:
     - Load toàn bộ UNet vào GPU (không chỉ attention layers)
     - Tính forward/backward qua toàn bộ UNet (chỉ update attention)
     - Xử lý prior preservation loss (class images) → tăng computation
     - Memory overhead cao hơn do phải giữ toàn bộ model
   - Không thể train lại nhiều lần để tối ưu hyperparameters
   - Kaggle weekly quota: Giới hạn số lần chạy GPU/week
   
4. **Resolution thấp**:
   - Input: 256×256 (thay vì 512×512) để tiết kiệm memory
   - Mất chi tiết texture và brushstrokes ở resolution thấp

### 2c. Fine-tune Textual Inversion

**Mục tiêu**: Học một embedding mới trong CLIP text encoder đại diện cho phong cách (`sks style`) thay vì fine-tune toàn bộ UNet.

**Cấu hình**:

- Base model: `runwayml/stable-diffusion-v1-5`
- Modules train: Textual embedding (768 chiều) dành cho token mới `<sks_style>`
- Learning rate: 5e-5
- Batch size: 1 (gradient accumulation 4)
- Steps: 400 per style
- Optimizer: AdamW
- Scheduler: Constant

**Yêu cầu thêm**:

- Captions chứa token đặc biệt (`<sks_style> style`)
- 20 instance images từ COCO dataset đã resize 512x512
- Theo dõi loss embedding để tránh overfit
- Embedding được scale về norm trung bình của vocabulary

**Kết quả**:

- Checkpoint embedding < 1MB/style (dễ chia sẻ)
- Có thể kết hợp với LoRA hoặc dùng riêng để generate ảnh theo phong cách

### 3. Inference

1. Encode Content_Image → latent vector (VAE encoder)
2. Load LoRA checkpoint tương ứng với style đã chọn
3. Áp dụng LoRA weights vào UNet
4. Denoise trong latent space với UNet
5. Decode → ảnh mới mang phong cách đã học (VAE decoder)

---

## Dataset

| Loại | Tên Dataset | Quy Mô | Ghi Chú |
|------|------------|--------|---------|
| **Content** | COCO 2017 | 118k train, 5k val | Ảnh thực tế đời thường |
| **Style** | WikiArt | 5 phong cách, 98-481 ảnh/phong cách | Tranh nghệ thuật |

---

## Phân Công Công Việc

### Nguyễn Khang Hy (2352662) - DreamBooth Training & Evaluation

**Trách nhiệm chính**:

- Quản lý dự án: Timeline, phân công, theo dõi tiến độ
- Tích hợp: Đảm bảo các phần code hoạt động cùng nhau
- Documentation: README, báo cáo cuối kỳ, presentation

**Công việc kỹ thuật**:

1. **EDA & Data Analysis**:
   - Phân tích dataset COCO và WikiArt
   - Thống kê phân phối, visualize samples
   - Identify potential issues

2. **DreamBooth Training**:
   - Fine-tune DreamBooth cho 2 phong cách nghệ thuật
   - **Chỉ train attention layers của UNet** (do hạn chế GPU memory trên Kaggle)
   - Tối ưu memory cho Kaggle GPU (CPU offloading, VAE slicing, attention slicing, resolution reduction)
   - Implement freeze/unfreeze logic để chỉ train attention layers
   - Hyperparameter tuning (learning rate, prior loss weight, steps)
   - Ghi nhận thời gian train, kích thước checkpoint, GPU usage
   - Save/load DreamBooth checkpoints (chỉ attention layers)

3. **Evaluation Framework**:
   - CLIP-Based metrics: CLIP-Content Similarity, CLIP-Style Similarity (sử dụng style centroid), Content Retention Rate
   - Load style reference images từ WikiArt (20 ảnh/phong cách)
   - Test set: 8 ảnh COCO val2017 (256×256) được cố định qua `content_paths.json`
   - So sánh LoRA vs DreamBooth vs Textual Inversion

4. **Results & Reporting**:
   - Tổng hợp kết quả training từ cả 3 phương pháp
   - So sánh các phong cách và các phương pháp fine-tuning
   - Viết báo cáo cuối kỳ

**Deliverables**:

- Notebook: `00-Data-EDA.ipynb`
- Notebook: `01b_DreamBooth_Training.ipynb`
- Notebook: `02b-Dreambooth-Inference-Test.ipynb`
- Notebook: `04a_Evaluation_Metrics_LoRA.ipynb`
- Notebook: `04b_Evaluation_Metrics_DreamBooth_TI.ipynb`
- Notebook: `05_Results_Analysis_FINALnew.ipynb`
- Trained DreamBooth checkpoints (2 styles: Contemporary_Realism, New_Realism)
- Evaluation report với CLIP-based metrics
- Slide

---

### Phan Đức Thành Phát (23521149) - LoRA Training

**Trách nhiệm chính**:

- Fine-tuning LoRA cho các phong cách nghệ thuật
- Tối ưu pipeline huấn luyện
- Hyperparameter tuning
- Cung cấp inference pipeline ổn định cho toàn hệ thống

**Công việc kỹ thuật**:

1. **LoRA Implementation**:
   - Implement LoRA layers cho UNet
   - Setup training pipeline với diffusers library
   - Loss function implementation (L2 + LPIPS + StyleLoss)

2. **Training & Optimization**:
   - Fine-tune LoRA cho 5 phong cách (Action_painting, Analytical_Cubism, Contemporary_Realism, New_Realism, Synthetic_Cubism)
   - Sử dụng 100 ảnh/phong cách từ WikiArt
   - Hyperparameter tuning (rank, learning rate, batch size)
   - Monitoring training progress
   - Save/load LoRA checkpoints

3. **Data Pipeline**:
   - Prepare training data (content-style pairs)
   - Data augmentation
   - DataLoader implementation

4. **Inference Support**:
   - Implement inference script và tối ưu tốc độ
   - Bàn giao checkpoints + hướng dẫn load LoRA cho pipeline chung
   - Hỗ trợ Minh Quốc tích hợp các lựa chọn LoRA trong demo

**Deliverables**:

- Notebook: `01a_LoRA_Training.ipynb`
- Notebook: `02a-LoRA-Inference-Test.ipynb`
- Trained LoRA checkpoints (5 styles: Action_painting, Analytical_Cubism, Contemporary_Realism, New_Realism, Synthetic_Cubism)
- Training logs và metrics
- Thuyết trình

---

### Trần Minh Quốc (MSSV) - Textual Inversion & Demo

**Trách nhiệm chính**:

- Fine-tuning textual inversion embeddings cho từng phong cách
- Phát triển demo Gradio tích hợp lựa chọn mô hình (LoRA / DreamBooth / Textual Inversion)
- Phối hợp inference pipeline để hỗ trợ nhiều baseline

**Công việc kỹ thuật**:

1. **Textual Inversion Training**:
   - Chuẩn bị instance captions với token đặc biệt `<sks_style>`
   - Huấn luyện embedding trên SD v1.5 (400 steps)
   - Sử dụng 20 ảnh từ COCO dataset
   - Quản lý checkpoint embeddings (.pt)
   - Ghi nhận thời gian train, kích thước checkpoint, GPU usage

2. **Demo & UX**:
   - Mở rộng notebook `03_Demo_Application.ipynb`
   - Demo Gradio hỗ trợ 5 LoRA styles với multi-adapter switching
   - Cho phép người dùng chọn style, điều chỉnh LoRA weight, denoise strength, guidance scale, steps, seed
   - Tích hợp inference pipeline cho LoRA
   - Xuất bản hướng dẫn sử dụng/demo video

3. **Inference Integration**:
   - Cập nhật `src/infer.py` để hỗ trợ textual inversion weights
   - Đảm bảo compatibility với LoRA và DreamBooth outputs
   - Hỗ trợ load và switch giữa các model types

**Deliverables**:

- Notebook: `01c_Textual_Inversion_Training.ipynb`
- Notebook: `02c-Textual-Inversion-Inference-Test.ipynb`
- Notebook: `03_Demo_Application.ipynb`
- Textual inversion embedding checkpoints (1 style: sks_style)
- Demo app (Gradio) + video/screenshots

---

## Cấu Trúc Thư Mục

```
customized-image-generation/
│
├── README.md                          # File mô tả toàn bộ dự án
├── .gitignore                         # Loại bỏ checkpoints, datasets, models
├── requirements.txt                   # Danh sách dependencies
│
├── notebooks/                                        # Nơi làm việc chính
│   ├── 00-Data-EDA.ipynb                             # EDA và phân tích dữ liệu (Hy)
│   ├── 01a_LoRA_Training.ipynb                       # LoRA training (Phát)
│   ├── 01b_DreamBooth_Training.ipynb                 # DreamBooth training (Hy)
│   ├── 01c_Textual_Inversion_Training.ipynb          # Textual inversion (Minh Quốc)
│   ├── 02a-LoRA-Inference-Test.ipynb                 # Test inference LoRA
│   ├── 02b-Dreambooth-Inference-Test.ipynb           # Test inference DreamBooth
│   ├── 02c-Textual-Inversion-Inference-Test.ipynb    # Test inference TI
│   ├── 03_Demo_Application.ipynb                     # Demo Gradio (Minh Quốc)
│   ├── 04a_Evaluation_Metrics_LoRA.ipynb             # Đánh giá LoRA (Hy)
│   ├── 04b_Evaluation_Metrics_DreamBooth_TI.ipynb    # Đánh giá DreamBooth + TI (Hy)
│   └── 05_Results_Analysis_FINALnew.ipynb            # Phân tích và so sánh kết quả (Hy)
│
│
├── docs/                              # Tài liệu chi tiết
│   ├── architecture.md                # Giải thích SD + LoRA
│   ├── training_guide.md              # Hướng dẫn training
│   └── evaluation_metrics.md          # Cách tính các chỉ số
│
└── results/                           # Kết quả mẫu
    ├── eda/                           # Kết quả EDA
    ├── metrics/                       # Metrics và logs
    ├── models/                        # Models
    └── samples/                       # Output samples
```

---

## Tech Stack & Tools

### Development Environment

- **Primary**: Kaggle Notebooks (GPU: P100/T4)
- **Datasets**: Kaggle Datasets (COCO 2017, WikiArt)
- **Version Control**: GitHub

### Core Libraries

```python
# Deep Learning
torch >= 2.0.0
torchvision >= 0.15.0
diffusers >= 0.21.0
transformers >= 4.30.0
accelerate >= 0.20.0

# Stable Diffusion
safetensors
peft  # LoRA implementation

# Computer Vision
opencv-python
Pillow
scikit-image

# Evaluation Metrics
pytorch-fid
lpips
torchmetrics

# Visualization
matplotlib
seaborn

# Demo
gradio >= 3.50.0

# Utils
numpy
pandas
tqdm
PyYAML
```

---

## Baseline và Chiến Lược Đánh Giá

### Baseline

**Baseline chính**: Stable Diffusion v1.5 gốc (`runwayml/stable-diffusion-v1-5`)

- Model đã được train sẵn, download từ Hugging Face (không train từ đầu)
- Sử dụng text prompt để generate ảnh
- Không có style transfer cụ thể
- Mục đích: So sánh để chứng minh LoRA fine-tuning cải thiện chất lượng

**Baseline fine-tuning 1**: LoRA (Low-Rank Adaptation)

- Train ~4-8M parameters, checkpoint 4-8MB
- Ưu tiên lightweight, dễ triển khai nhiều style
- Training nhanh, memory efficient

**Baseline fine-tuning 2**: DreamBooth

- **Chỉ fine-tune attention layers của UNet** (do hạn chế GPU memory trên Kaggle)
- Train ~30% parameters (~260M thay vì 860M full UNet)
- Instance images: 40 ảnh/phong cách
- Class images: 200 ảnh/phong cách (prior preservation)
- Checkpoint: Chỉ lưu attention layers đã train (nhỏ hơn full model)
- Training lâu hơn (~12 giờ), memory usage ~5-6GB VRAM (với optimizations)
- Sử dụng prior preservation loss (weight=0.6) để tránh overfitting
- **Lưu ý**: Trong implementation này, không train full UNet do hạn chế phần cứng

**Baseline fine-tuning 3**: Textual Inversion

- Fine-tune embedding của token đặc biệt `<sks_style>` trong CLIP text encoder (~768 params)
- Instance images: 20 ảnh từ COCO dataset
- Checkpoint < 1MB, training 400 steps, phù hợp cho Kaggle
- Rất nhẹ, training nhanh nhất
- Embedding được scale về norm trung bình của vocabulary

**Baseline tham khảo**: Stable Diffusion v1.5 gốc (không fine-tune)

Xem chi tiết tại: [`docs/baseline_and_evaluation.md`](docs/baseline_and_evaluation.md)

### Model Training

**Base Model**:

- Download từ Hugging Face: `runwayml/stable-diffusion-v1-5`
- **KHÔNG train từ đầu**, chỉ download và sử dụng
- Cấu trúc: VAE (~85M) + UNet (~860M) + CLIP (~123M, không dùng)

**LoRA Fine-Tuning** (Phát):

- Load base model SD v1.5
- Thêm LoRA layers vào UNet attention layers
- **CHỈ train LoRA weights** (~4-8M params), không train toàn bộ UNet
- Train trên style images từ WikiArt
- Mỗi style → 1 LoRA checkpoint (~4-8MB)

**DreamBooth Fine-Tuning** (Hy):

- Load base model SD v1.5
- **Chỉ fine-tune attention layers của UNet** (do hạn chế GPU memory trên Kaggle)
  - Freeze tất cả parameters của UNet
  - Chỉ enable gradient cho attention layers (cross-attention và self-attention)
  - Train ~30% parameters (~260M thay vì 860M)
- Sử dụng prior preservation với class images để tránh overfitting
- Train trên instance images + class images từ WikiArt
- Memory optimizations: CPU offloading, VAE slicing, attention slicing, resolution 256
- Mỗi style → 1 DreamBooth checkpoint (chỉ lưu attention layers đã train)

**Textual Inversion Fine-Tuning** (Minh Quốc):

- Load base model SD v1.5
- Fine-tune embedding của special token trong CLIP text encoder
- Train trên style images với captions chứa special token
- Mỗi style → 1 embedding checkpoint (< 1MB)

**Hyperparameters** (tham khảo):

- LoRA: Rank=4, LR=1e-4, Batch=2, Steps=1.5k, Resolution=512
- DreamBooth: LR=1e-5, Batch=1 (gradient accumulation=16), Steps=2k, Prior loss weight=0.6, Resolution=256
- Textual Inversion: LR=5e-5, Batch=1 (gradient accumulation=4), Steps=400, Resolution=512

### Evaluation Strategy (Similarity Version)

**Metrics sử dụng (Similarity Version)**:

- **CLIP-content Similarity**: `cos_sim(clip(output), clip(content))` – Đo độ tương đồng ngữ nghĩa với ảnh gốc. Giá trị gần **1.0** là tốt (giữ nội dung tốt).
- **CLIP-style Similarity**: `cos_sim(clip(output), style_centroid)` – Đo độ tương đồng với vector trung bình (centroid) của tập ảnh style. `style_centroid = mean([clip(style_i)])` trên 20 ảnh tham chiếu phong cách. Giá trị gần **1.0** là tốt (giống style mẫu).
- **Content Retention Rate** (trước đây Style Strength): `CLIP-content Similarity / baseline_CLIP-content Similarity` – Tỷ lệ giữ nội dung so với baseline. Quanh **1.0**: cân bằng; <1: hy sinh nội dung mạnh; >1: đôi khi baseline bị nhiễu.

**Additional Metrics**:

- **Inference Time**: < 5s/ảnh trên Kaggle P100/T4.

**Test Set**:

- **Content**: Tập con COCO val2017 (8 ảnh resized 256×256). Danh sách ảnh được cố định và chia sẻ giữa mọi notebook qua `content_paths.json`.
- **Style**: WikiArt images (20 ảnh/style). Danh sách ảnh cố định qua `style_paths.json` để LoRA và DreamBooth/TI dùng chung baseline.

**So sánh với Baseline**:

- Baseline: `runwayml/stable-diffusion-v1-5` chạy img2img cùng content images (dùng để chuẩn hóa Style Strength Score).
- DreamBooth: Contemporary_Realism, New_Realism (2 styles).
- LoRA: Action_painting, Analytical_Cubism, Contemporary_Realism, New_Realism, Synthetic_Cubism (5 styles).
- Textual Inversion: sks_style (1 style, sử dụng Contemporary_Realism làm style reference cho evaluation).

**Nguyên lý đánh giá (Similarity)**:

- **Content Preservation**: CLIP-content Similarity càng gần hoặc ≥ baseline càng tốt. Content Retention Rate ~ 0.90–1.05 là chấp nhận được.
- **Style Quality**: CLIP-style Similarity cao phản ánh phong cách được học ổn định (so với style centroid từ 20 ảnh reference).
- **Trade-off**: CLIP-style Similarity cao nhưng Retention Rate <0.9 ⇒ phong cách mạnh làm mất nội dung. LoRA thường cân bằng cả hai tốt nhất; DreamBooth (attention-only) đôi lúc giảm content similarity; Textual Inversion vừa phải cả hai chiều.

---

## Evaluation Metrics (Similarity)

### Target Metrics

- **CLIP-content Similarity**: Giá trị gần **1.0** là tốt (giữ nội dung tốt). Target ≥ (baseline − 0.05) hoặc Retention Rate ≥ 0.90.
- **Content Retention Rate**: 0.90 – 1.05 (≈1.0 cân bằng giữa content & style).
- **CLIP-style Similarity**: Giá trị gần **1.0** là tốt (giống style mẫu). Target > 0.50 (thực nghiệm) ⇒ phong cách được học tốt; càng cao càng thể hiện đặc trưng rõ.
- **Inference time < 5s/ảnh** với 256×256 trên Kaggle P100/T4.

### Lưu ý về Trade-off (Similarity)

- **LoRA**: Thường đạt content similarity cao (0.57-0.58) và style similarity cao (0.64-0.74) ⇒ cân bằng tốt nhất.
- **DreamBooth (attention-only)**: Có thể đạt style similarity vừa phải (0.69-0.70) nhưng đôi khi giảm content similarity (0.56-0.57) ⇒ cần tinh chỉnh thêm nếu muốn cân bằng.
- **Textual Inversion**: Nhanh, nhẹ; similarity trung bình cả hai chiều (content: 0.57, style: 0.67) ⇒ phù hợp cho mở rộng nhanh phong cách.
- Sử dụng style centroid (20 ảnh reference) giảm biến thiên giữa các cặp so sánh, giúp so sánh công bằng giữa phương pháp.

---

## Tài Liệu Tham Khảo

### Papers

1. **Stable Diffusion**: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
2. **LoRA**: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
3. **Style Transfer**: [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)

### Implementation References

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [PEFT Library](https://github.com/huggingface/peft)
- [LoRA for Stable Diffusion](https://huggingface.co/docs/peft/task_guides/stable_diffusion)

### Datasets

- [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)
- [WikiArt](https://www.kaggle.com/datasets/steubk/wikiart)

---

## Liên Hệ

- **GitHub**: https://github.com/HyIsNoob/customized-image-generation
- **Issues**: Sử dụng GitHub Issues để báo lỗi và đề xuất
