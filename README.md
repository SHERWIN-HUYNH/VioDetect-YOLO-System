🚗 Intelligent Traffic Monitoring and Automatic Vehicle Counting System
📝 Giới thiệu
Dự án tập trung vào việc xây dựng một hệ thống Giám sát Giao thông Thông minh (ITS) có khả năng tự động phát hiện, theo dõi và đếm phương tiện giao thông từ luồng video thời gian thực. Điểm đặc biệt của hệ thống là khả năng nhận diện biển số xe (LPR) tích hợp module hậu xử lý thông minh để giảm thiểu sai sót OCR.

✨ Tính năng nổi bật

Phát hiện đa đối tượng: Sử dụng kiến trúc YOLOv11 (phiên bản Nano và Small) để nhận diện phương tiện và biển số với độ chính xác cao.


Theo dõi đối tượng (MOT): Tích hợp ByteTrack để duy trì ID phương tiện ổn định, ngay cả khi bị che khuất một phần.


Xử lý bất đồng bộ (Asynchronous Pipeline): Kiến trúc đa luồng (multi-threading) với hàng đợi luồng an toàn (thread-safe queues) giúp loại bỏ "nút thắt cổ chai" khi chạy các mô hình OCR nặng.


Hậu xử lý thông minh (Smart Heuristics): Module sửa lỗi dựa trên quy tắc cú pháp biển số xe Việt Nam (ví dụ: tự động sửa lỗi nhầm lẫn giữa '8' và 'B').

🏗 Kiến trúc hệ thống
Hệ thống được thiết kế theo dạng pipeline bất đồng bộ để tối ưu hóa hiệu suất:

Mô-đun Phát hiện: YOLOv11 thực hiện nhận diện phương tiện và vùng chứa biển số.

Mô-đun Theo dõi: ByteTrack gán ID và theo dõi quỹ đạo di chuyển của xe.


Mô-đun LPR (License Plate Recognition): Sử dụng PaddleOCR kết hợp với logic căn chỉnh vùng nhìn để trích xuất ký tự.

Mô-đun Phân tích: Đếm xe tự động dựa trên vạch ranh giới ảo và lưu trữ kết quả.

🛠 Tech Stack

Ngôn ngữ: Python 3.8.10.


Framework AI: Ultralytics YOLOv11, ByteTrack, PaddleOCR.


Thư viện bổ trợ: OpenCV, NumPy, Matplotlib, Pandas, Scikit-learn.


Hạ tầng: Docker, Tesla T4 GPU (16GB VRAM).

📊 Kết quả thực nghiệm
Hệ thống đã đạt được những con số ấn tượng trong môi trường thử nghiệm:


Hiệu năng xử lý: Tăng tốc độ khung hình từ 6.5 FPS (tuần tự) lên 18.2 FPS (bất đồng bộ) — tăng trưởng 180% throughput.


Tối ưu tài nguyên: Giảm tải CPU/GPU lên tới 60% nhờ giải thuật "Smart Stop" (dừng xử lý khi biển số đã được xác nhận).

Độ chính xác:

mAP@0.5 cho nhận diện phương tiện đạt 77.12%.

Độ chính xác nhận diện biển số (LPR) đạt mức tin cậy cao nhờ module hậu xử lý.

🚀 Hướng phát triển
Tích hợp nhận diện hành vi vi phạm giao thông (vượt đèn đỏ, lấn làn).

Triển khai mô hình trên các thiết bị Edge (NVIDIA Jetson) để giám sát tại chỗ.

Nâng cấp lên kiến trúc Transformer-based để tăng độ chính xác trong điều kiện ánh sáng yếu.

👥 Tác giả

Huỳnh Chí Trung - Sinh viên ngành Trí tuệ Nhân tạo, PTIT.

Ths. Huỳnh Trung Trụ - Giảng viên hướng dẫn.
