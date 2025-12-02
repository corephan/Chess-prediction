# ♟️ Chess PGN Preprocessing Pipeline

Đây là tài liệu hướng dẫn và phân tích kỹ thuật cho quy trình tiền xử lý dữ liệu cờ vua (PGN) nhằm tạo ra một bộ dữ liệu cân bằng (Balanced Dataset) và tối ưu hóa cho mô hình học sâu (Deep Learning Model) dự đoán kết quả ván cờ (Game Outcome Prediction).

Quy trình này gồm 3 bước chính, được thiết kế để giải quyết triệt để các vấn đề về **Mất cân bằng dữ liệu** và **Rò rỉ dữ liệu**.

---

## 1. 📂 Cấu trúc Thư mục Dự kiến

Quy trình này giả định cấu trúc thư mục sau đây được thiết lập trong môi trường chạy:

```
.
├── PreProcessing/
│   ├── drawExtraction.py
│   ├── FEN_Splitting.py
│   └── PGN_to_Tensor.ipynb
└── DataSets/
    └── pgnData/
        ├── (1).pgn                      # File PGN chính
        ├── ...                           # Các file PGN khác
        └── all_draws_combined.pgn        # OUTPUT từ drawExtraction.py
    └── pgnData (Balanced)/
        ├── train.pgn                     # OUTPUT từ FEN_Splitting.py (80% ván)
        ├── validation.pgn                # OUTPUT từ FEN_Splitting.py (10% ván)
        └── test.pgn                      # OUTPUT từ FEN_Splitting.py (10% ván)
```

---

## 2. 📝 Phân tích Kỹ thuật các Module

### 2.1. `drawExtraction.py`: Thu thập Dữ liệu Ván hòa

**Mục đích**  
Lọc và tổng hợp các ván cờ có kết quả hòa (`1/2-1/2`) từ nhiều file PGN nguồn.

**Output**  
`all_draws_combined.pgn` trong thư mục `DataSets/pgnData`.

**Công nghệ**  
Sử dụng thư viện `python-chess` để đọc PGN.

**Ghi chú**  
Thực hiện xóa file output cũ nếu tồn tại trước khi ghi mới.

---

### 2.2. `FEN_Splitting.py`: Cân bằng và Phân chia Tập

Module này giải quyết các thách thức cốt lõi về chất lượng dữ liệu bằng cách xử lý ở cấp độ **Ván cờ (Game-Level)**.

#### 🔑 Giải quyết Vấn đề Data Imbalance & Leakage

| Vấn đề             | Giải pháp Kỹ thuật                                                                                                                                                            | Phân tích cho AI Engineer                                                                                                                                                            |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Data Imbalance** | Lấy mẫu ngẫu nhiên (Sampling): Lấy mẫu tối đa 100,000 ván (`TARGET_PER_CLASS`) cho mỗi lớp kết quả (`1-0`, `0-1`, `1/2-1/2`) từ các file PGN nguồn.                           | Đảm bảo mô hình được huấn luyện trên các lớp có phân bố đồng đều, dẫn đến khả năng dự đoán kết quả ván cờ cân bằng hơn.                                                              |
| **Data Leakage**   | Phân chia Game-Level: Vị trí của mỗi ván cờ được đánh dấu bằng `offset` và `length`, sau đó được gán ngẫu nhiên vào duy nhất một tập (Train/Valid/Test) trước khi trích xuất. | Ngăn chặn rò rỉ dữ liệu bằng cách đảm bảo mô hình không nhìn thấy bất kỳ trạng thái (FEN) nào từ ván cờ kiểm thử trong quá trình huấn luyện, giữ cho kết quả đánh giá là khách quan. |

#### Chi tiết Kỹ thuật

- **Tối ưu Đọc file:** Sử dụng Memory Mapping (`mmap`) để quét file PGN siêu tốc và tìm offset của từng ván cờ, tối ưu hóa I/O.
- **Tỷ lệ Phân chia:** Train (0.8), Validation (0.1), Test (0.1).
- **Tối ưu Ghi file:** Ghi file PGN kết quả bằng Binary Write với Buffer 1MB để tăng tốc độ ghi.

---

### 2.3. `PGN_to_Tensor.ipynb`: Mã hóa Trạng thái Cờ vua

Notebook này chuyển đổi PGN thành Tensor, là định dạng số hóa sẵn sàng cho quá trình huấn luyện mô hình Học sâu.

#### Cấu trúc Dữ liệu Đầu vào (Feature Engineering)

**Input Tensor (X)** có kích thước $\mathbf{(8, 8, 41)}$.

| Đặc trưng (Kênh) | Phạm vi | Mô tả                                                                      |
| ---------------- | ------- | -------------------------------------------------------------------------- |
| 0-11             | 12 kênh | Mã hóa vị trí các quân cờ (6 loại x 2 màu).                                |
| 12-17            | 6 kênh  | Lượt đi, Quyền nhập thành, En Passant.                                     |
| 18-19            | 2 kênh  | Đồng hồ luật 50 nước và Số nước đi.                                        |
| 21-22            | 2 kênh  | Bản đồ các ô bị Tấn công (White/Black Attack).                             |
| 34-35            | 2 kênh  | Tốt thông (Passed Pawns) cho Trắng và Đen.                                 |
| 40               | 1 kênh  | Độ căng thẳng (Tension): Vùng giao thoa giữa các đòn tấn công của hai bên. |

**Output Label (y)** có kích thước $\mathbf{(3,)}$ (One-Hot Encoding cho 3 lớp: White Win, Black Win, Draw).

#### Chiến lược Lấy mẫu FEN (Sampling)

- **Lọc Khai cuộc:** Bỏ qua 20 nước đi đầu tiên của mỗi ván cờ để tập trung vào các trạng thái chiến thuật phức tạp hơn (Trung cuộc/Tàn cuộc).
- **Giảm tương quan:** Chỉ lấy mẫu mỗi 5 nước đi tiếp theo để giảm thiểu sự tương quan giữa các trạng thái liên tiếp.

#### Tối ưu hóa Hiệu năng

- **Đa luồng:** Sử dụng `multiprocessing.Pool` và `tqdm` để xử lý song song việc đọc PGN và mã hóa Tensor.
- **Sharding:** Xuất dữ liệu dưới dạng các file nén NumPy (`.npz`) (shards), với mỗi shard chứa 40,000 mẫu (`shard_size=40000`), phục vụ cho việc tải dữ liệu hiệu quả trong quá trình huấn luyện mô hình.
