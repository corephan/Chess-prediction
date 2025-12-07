import chess.pgn
import os
import time

def extract_all_draws(source_folder, source_files, output_filename):
    """
    Hàm đọc danh sách các file PGN, lọc ván hoà và ghi vào 1 file duy nhất.
    """
    # Tạo đường dẫn file output
    output_path = os.path.join(source_folder, output_filename)
    
    # Xoá file output cũ nếu tồn tại để ghi mới từ đầu
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"🗑️ Đã xoá file cũ: {output_filename}")

    total_draws = 0
    start_time = time.time()
    
    print(f"🚀 BẮT ĐẦU EXTRACT DRAW TỪ {len(source_files)} FILE...")
    print("-" * 60)

    # Mở file output ở chế độ 'a' (append - nối đuôi)
    # Dùng encoding='utf-8' để tránh lỗi ký tự lạ
    with open(output_path, 'w', encoding='utf-8') as pgn_out:
        
        for filename in source_files:
            file_path = os.path.join(source_folder, filename)
            
            if not os.path.exists(file_path):
                print(f"⚠️ Cảnh báo: Không tìm thấy file {filename}. Bỏ qua.")
                continue

            print(f"📂 Đang xử lý: {filename} ...")
            
            # Mở file input
            with open(file_path, 'r', encoding='utf-8') as pgn_in:
                file_draws = 0
                game_count = 0
                
                while True:
                    try:
                        # Đọc headers trước để check kết quả (Nhanh hơn đọc full game)
                        # Tuy nhiên để ghi ra file output, ta cần đọc full game.
                        # python-chess read_game sẽ đọc cả headers và moves.
                        game = chess.pgn.read_game(pgn_in)
                    except Exception as e:
                        # Bỏ qua các ván lỗi định dạng
                        continue
                    
                    if game is None:
                        break # Hết file
                    
                    game_count += 1
                    
                    # Check kết quả Hoà
                    result = game.headers.get("Result", "*")
                    if result == "1/2-1/2":
                        # Ghi ván cờ vào file output
                        print(game, file=pgn_out, end="\n\n")
                        file_draws += 1
                        total_draws += 1
                        
                        # In tiến độ mỗi 1000 ván hoà tìm được
                        if file_draws % 1000 == 0:
                            print(f"   -> Đã tìm thấy {file_draws} ván hoà trong file này...", end='\r')

            print(f"✅ Xong file {filename}. Tìm thấy: {file_draws} ván hoà.")

    # Tổng kết
    duration = time.time() - start_time
    print("-" * 60)
    print(f"🎉 HOÀN TẤT!")
    print(f"📊 Tổng số ván hoà thu được: {total_draws}")
    print(f"💾 File lưu tại: {output_path}")
    print(f"⏱️ Thời gian chạy: {duration:.2f} giây")

# --- CẤU HÌNH & CHẠY ---
if __name__ == "__main__":
    # Đường dẫn tới thư mục DataSets/pgnData
    BASE_DIR = "/content/ChessOutcomesPrediction/DataSets/pgnData"
    
    # Kiểm tra nếu thư mục không tồn tại
    if not os.path.exists(BASE_DIR):
        print(f"❌ Không tìm thấy thư mục: {BASE_DIR}")
        print("Vui lòng kiểm tra đường dẫn thư mục DataSets/pgnData")
        exit(1)

    # Tự động tìm tất cả file .pgn trong thư mục
    all_pgn_files = [f for f in os.listdir(BASE_DIR) if f.endswith('.pgn')]
    
    if len(all_pgn_files) == 0:
        print(f"❌ Không tìm thấy file .pgn nào trong thư mục: {BASE_DIR}")
        exit(1)
    
    # Sắp xếp và lấy 3 file đầu tiên (hoặc tất cả nếu ít hơn 3)
    SOURCE_FILES = sorted(all_pgn_files)[:3]
    
    print(f"📁 Thư mục: {BASE_DIR}")
    print(f"📄 Files được xử lý: {SOURCE_FILES}")
    print()
    
    # Tên file kết quả - lưu trong cùng thư mục pgnData
    OUTPUT_FILE = 'all_draws_combined.pgn'
    
    extract_all_draws(BASE_DIR, SOURCE_FILES, OUTPUT_FILE)
