# 🚗 Intelligent Traffic Monitoring and Automatic Vehicle Counting System

## 📝 Giới thiệu
[cite_start]Đồ án tốt nghiệp chuyên ngành Trí tuệ nhân tạo (2021-2026). Hệ thống được thiết kế để tự động hóa quy trình giám sát giao thông, bao gồm phát hiện phương tiện, theo dõi quỹ đạo và đếm xe thông minh từ luồng video thực tế.

## ✨ Tính năng nổi bật
* [cite_start]**Phát hiện đa đối tượng:** Sử dụng kiến trúc **YOLOv11** (Nano/Small) tối ưu cho tốc độ và độ chính xác.
* [cite_start]**Theo dõi đối tượng (MOT):** Tích hợp thuật toán **ByteTrack** để duy trì định danh (ID) phương tiện ổn định.
* [cite_start]**Xử lý bất đồng bộ:** Kiến trúc **Asynchronous Pipeline** giúp tăng hiệu suất xử lý lên 180% so với phương pháp tuần tự.
* [cite_start]**Hậu xử lý thông minh:** Tự động sửa lỗi OCR dựa trên quy tắc cú pháp biển số xe Việt Nam (ví dụ: sửa nhầm lẫn giữa '8' và 'B').

## 🏗 Kiến trúc hệ thống
Hệ thống được tổ chức thành các mô-đun xử lý chuyên biệt:
1. [cite_start]**Mô-đun Detection:** Nhận diện phương tiện và vị trí biển số xe.
2. [cite_start]**Mô-đun Tracking:** Theo dõi hành vi và gán ID duy nhất cho mỗi xe qua các khung hình.
3. [cite_start]**Mô-đun OCR & Counting:** Nhận dạng ký tự biển số và đếm xe dựa trên vạch ranh giới ảo.



## 🛠 Công nghệ sử dụng
* [cite_start]**Mô hình AI:** YOLOv11 (Ultralytics), ByteTrack, PaddleOCR.
* [cite_start]**Ngôn ngữ & Thư viện:** Python 3.8.10, OpenCV, NumPy, Pandas.
* [cite_start]**Phần cứng thử nghiệm:** NVIDIA Tesla T4 GPU (16GB VRAM).

## 📊 Kết quả thực nghiệm
* [cite_start]**Tốc độ xử lý:** Đạt **18.2 FPS** trên luồng dữ liệu thực tế (tăng từ 6.5 FPS ban đầu).
* [cite_start]**Độ chính xác:** mAP@0.5 cho nhận diện phương tiện đạt **77.12%**.
* [cite_start]**Tối ưu hóa:** Giảm tải hệ thống 60% nhờ thuật toán **"Smart Stop"** (ngừng xử lý vùng biển số khi đã đạt độ tin cậy).

## 👥 Thông tin tác giả
* [cite_start]**Sinh viên:** Huỳnh Chí Trung - MSSV: N21DCCN191.
* [cite_start]**Lớp:** E21CQCNTT01-N - Chuyên ngành: Trí tuệ nhân tạo.
* **Giảng viên hướng dẫn:** ThS. [cite_start]Huỳnh Trung Trụ.
