📁 System Module – Thiết kế và Mục đích

Thư mục System được xây dựng nhằm tách biệt các thành phần cốt lõi của hệ thống xử lý dữ liệu và mô hình học máy, giúp dự án có cấu trúc rõ ràng, dễ mở rộng và dễ bảo trì. Việc tổ chức mã nguồn theo module thay vì đặt toàn bộ trong notebook giúp đảm bảo tính tái sử dụng, tính nhất quán và phù hợp với chuẩn triển khai các hệ thống Machine Learning trong thực tế.

Cụ thể, các file trong thư mục System đảm nhiệm những vai trò sau:

data_extraction.py :
Thực hiện bước data extraction, chịu trách nhiệm đọc các file PGN gốc và lọc ra các ván cờ theo tiêu chí nhất định (ví dụ: các ván hòa). Việc tách riêng module này giúp quá trình thu thập dữ liệu độc lập với các bước xử lý và huấn luyện mô hình phía sau.

process_merge.py :
Đảm nhiệm bước data preprocessing và dataset construction, bao gồm việc cân bằng số lượng mẫu giữa các lớp (White win, Black win, Draw) và chia dữ liệu thành các tập train / validation / test. Module này giúp tránh hiện tượng data leakage và đảm bảo tính công bằng của bộ dữ liệu dùng để huấn luyện mô hình.

utils.py :
Cung cấp các hàm tiện ích dùng chung cho toàn bộ hệ thống, tiêu biểu là hàm chuyển đổi trạng thái bàn cờ (chess.Board) sang vector số 64 phần tử. Việc đặt các hàm này trong utils giúp tránh trùng lặp code và đảm bảo cùng một cách mã hóa được sử dụng xuyên suốt pipeline.

train_model.py :
Chịu trách nhiệm xây dựng và lưu mô hình TensorFlow. Việc tách riêng module huấn luyện giúp quá trình training và inference có thể được thực hiện độc lập với notebook demo, đồng thời cho phép tái sử dụng mô hình đã huấn luyện trong các bước đánh giá hoặc triển khai sau này.

models/ :
Thư mục lưu trữ các mô hình đã được huấn luyện và lưu dưới dạng .keras. Cách tổ chức này phù hợp với chuẩn lưu trữ mô hình trong các hệ thống Machine Learning, cho phép tải lại mô hình mà không cần huấn luyện lại từ đầu.

Nhìn chung, thư mục System đóng vai trò như lõi xử lý (core system) của dự án, trong khi notebook demo chỉ đóng vai trò minh họa và thử nghiệm. Cách tổ chức này giúp dự án có tính học thuật cao hơn, dễ kiểm soát quy trình xử lý dữ liệu, và sẵn sàng mở rộng sang các mô hình phức tạp hơn trong tương lai.
