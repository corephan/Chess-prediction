import os
import random
import mmap
import gc
from tqdm import tqdm
import time

# =================================================================
# 🎛️ CẤU HÌNH HỆ THỐNG
# =================================================================
# Lấy đường dẫn thư mục gốc của project (lùi lên 1 cấp từ PreProcessing)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn file đã được sửa lỗi cú pháp:
# Sử dụng os.path.join để kết hợp các thành phần đường dẫn.
FILE_MAIN = os.path.join(BASE_DIR, 'DataSets', 'pgnData', '(1).pgn')
FILE_DRAW = os.path.join(BASE_DIR, 'DataSets', 'pgnData', 'all_draws_combined.pgn')
OUTPUT_DIR = os.path.join(BASE_DIR, 'DataSets', 'pgnData (Balanced)')

# Chỉ tiêu (Số lượng ván mỗi lớp)
TARGET_PER_CLASS = 100000
RATIOS = (0.8, 0.1, 0.1)

def scan_pgn_indices_fast(file_path):
    """Quét file siêu tốc dùng mmap"""
    if not os.path.exists(file_path):
        print(f"❌ Lỗi: Không tìm thấy {file_path}")
        # Trả về kết quả rỗng để tránh lỗi tiếp theo
        return {'1-0': [], '0-1': [], '1/2-1/2': []} 

    indices = {'1-0': [], '0-1': [], '1/2-1/2': []}
    file_size = os.path.getsize(file_path)
    
    print(f"🔍 Đang đánh chỉ mục: {os.path.basename(file_path)} ({file_size/1024/1024:.1f} MB)...")
    
    try:
        with open(file_path, "rb") as f:
            # Dùng mmap để truy cập file như RAM
            # Thêm try-except block cho mmap nếu file quá lớn hoặc có vấn đề về quyền truy cập
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                cursor = 0
                # Pre-compile các chuỗi bytes để so sánh nhanh hơn
                KEY_EVENT = b"[Event"
                RES_W = b'[Result "1-0"]'
                RES_L = b'[Result "0-1"]'
                RES_D = b'[Result "1/2-1/2"]'
                
                # Bắt đầu thanh progress
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Indexing") as pbar:
                    
                    # Tìm ván đầu tiên
                    start_pos = mm.find(KEY_EVENT, cursor)
                    while start_pos != -1:
                        # Tìm ván tiếp theo
                        next_pos = mm.find(KEY_EVENT, start_pos + 1)
                        
                        # Tính độ dài ván cờ
                        if next_pos == -1:
                            length = file_size - start_pos
                        else:
                            length = next_pos - start_pos
                        
                        # Đọc Header (tối đa 2048 bytes đầu) để tìm kết quả
                        # Một số PGN có header dài; tăng ngưỡng để bền hơn
                        head = mm[start_pos : start_pos + min(length, 2048)]
                        
                        # Check nhanh
                        if RES_W in head:
                            indices['1-0'].append((start_pos, length))
                        elif RES_L in head:
                            indices['0-1'].append((start_pos, length))
                        elif RES_D in head:
                            indices['1/2-1/2'].append((start_pos, length))
                        
                        # Cập nhật thanh progress
                        processed = (next_pos if next_pos != -1 else file_size) - cursor
                        pbar.update(processed)
                        
                        # Cập nhật con trỏ cho lần lặp tiếp theo
                        cursor = next_pos if next_pos != -1 else file_size
                        start_pos = next_pos
    except Exception as e:
        print(f"❌ Lỗi trong quá trình đánh chỉ mục: {e}")
        return {'1-0': [], '0-1': [], '1/2-1/2': []}
        
    return indices

def process_and_merge():
    # Thêm kiểm tra file đầu vào trước khi chạy
    if not (os.path.exists(FILE_MAIN) and os.path.exists(FILE_DRAW)):
        print("🚨 Lỗi: Không tìm thấy ít nhất một trong hai file nguồn. Vui lòng kiểm tra lại đường dẫn.")
        print(f"  MAIN: {FILE_MAIN}")
        print(f"  DRAW: {FILE_DRAW}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. SCAN DỮ LIỆU
    idxs_main = scan_pgn_indices_fast(FILE_MAIN)
    pool_w = idxs_main['1-0']
    pool_l = idxs_main['0-1']
    # Giải phóng memory ngay lập tức cho phần không dùng
    del idxs_main
    gc.collect() 
    
    idxs_draw = scan_pgn_indices_fast(FILE_DRAW)
    pool_d = idxs_draw['1/2-1/2']
    del idxs_draw
    gc.collect()

    print(f"\n📊 KHO DỮ LIỆU:")
    print(f"   - White Wins: {len(pool_w)}")
    print(f"   - Black Wins: {len(pool_l)}")
    print(f"   - Draws:      {len(pool_d)}")

    # 2. SAMPLING (LẤY MẪU)
    # Hàm lấy mẫu ngẫu nhiên an toàn
    def safe_sample(pool, n, label):
        if len(pool) < n:
            print(f"⚠️ Cảnh báo: {label} chỉ có {len(pool)} (Cần {n}). Lấy tất cả.")
            # Tạo bản sao trước khi xáo trộn để không làm thay đổi pool gốc
            sampled = pool[:] 
            random.shuffle(sampled)
            return sampled
        return random.sample(pool, n)

    final_w = safe_sample(pool_w, TARGET_PER_CLASS, "White")
    final_l = safe_sample(pool_l, TARGET_PER_CLASS, "Black")
    final_d = safe_sample(pool_d, TARGET_PER_CLASS, "Draw")
    
    # Dọn dẹp pool gốc để nhẹ RAM
    del pool_w, pool_l, pool_d
    gc.collect()

    # 3. SPLIT (CHIA TẬP)
    def split_data(lst):
        n = len(lst)
        n1 = int(n * RATIOS[0])
        n2 = int(n * RATIOS[1])
        # Đảm bảo tổng số lượng không vượt quá n
        return lst[:n1], lst[n1:n1+n2], lst[n1+n2:n] 

    w_sets = split_data(final_w)
    l_sets = split_data(final_l)
    d_sets = split_data(final_d)

    # Gắn thẻ nguồn gốc: (offset, length, file_source)
    def tag(lst, src): 
        # Gắn đúng nguồn file đã truyền vào (FILE_MAIN hoặc FILE_DRAW)
        # Tránh heuristic theo tên file vì dễ sai lệch
        return [(x[0], x[1], src) for x in lst]

    # Tagging cần dùng đường dẫn đã được chuẩn hóa (FILE_MAIN/FILE_DRAW)
    train_items = tag(w_sets[0], FILE_MAIN) + tag(l_sets[0], FILE_MAIN) + tag(d_sets[0], FILE_DRAW)
    valid_items = tag(w_sets[1], FILE_MAIN) + tag(l_sets[1], FILE_MAIN) + tag(d_sets[1], FILE_DRAW)
    test_items = tag(w_sets[2], FILE_MAIN) + tag(l_sets[2], FILE_MAIN) + tag(d_sets[2], FILE_DRAW)

    # Shuffle lần cuối
    random.shuffle(train_items)
    random.shuffle(valid_items)
    random.shuffle(test_items)
    
    # In ra số lượng cuối cùng để xác nhận
    print(f"\n✨ TỔNG KẾT BỘ DỮ LIỆU CÂN BẰNG:")
    print(f"   - Train: {len(train_items)} ({len(w_sets[0])} W / {len(l_sets[0])} B / {len(d_sets[0])} D)")
    print(f"   - Valid: {len(valid_items)} ({len(w_sets[1])} W / {len(l_sets[1])} B / {len(d_sets[1])} D)")
    print(f"   - Test:  {len(test_items)} ({len(w_sets[2])} W / {len(l_sets[2])} B / {len(d_sets[2])} D)")

    del w_sets, l_sets, d_sets, final_w, final_l, final_d
    gc.collect()

    # 4. WRITING (GHI FILE TỐI ƯU BUFFER)
    print("\n🚀 Đang ghi file...")
    
    # Mở file nguồn 1 lần duy nhất
    f_main = open(FILE_MAIN, 'rb')
    f_draw = open(FILE_DRAW, 'rb')

    def write_dataset(filename, items):
        path = os.path.join(OUTPUT_DIR, filename)
        print(f"💾 Ghi {filename} ({len(items)} ván)...")
        
        # Buffering = 1MB (Tối ưu cho tốc độ ghi đĩa)
        with open(path, 'wb', buffering=1024*1024) as f_out:
            for start, length, src_file in tqdm(items):
                # Chọn file handle đúng (sử dụng so sánh chính xác)
                handle = f_main if src_file == FILE_MAIN else f_draw
                
                # Việc seek và read này là điểm mấu chốt của tối ưu tốc độ đọc file phân tán
                handle.seek(start)
                data = handle.read(length)
                f_out.write(data)
                
                # Đảm bảo có hai dòng trống sau mỗi ván cờ (chuẩn PGN)
                if not data.endswith(b"\n\n"):
                    f_out.write(b"\n\n")

    write_dataset('train.pgn', train_items)
    write_dataset('validation.pgn', valid_items)
    write_dataset('test.pgn', test_items)

    f_main.close()
    f_draw.close()
    print("\n✅ HOÀN TẤT TUYỆT ĐỐI!")

if __name__ == "__main__":
    process_and_merge()