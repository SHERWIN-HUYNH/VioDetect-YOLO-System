from ultralytics import YOLO

# 1. Load model của bạn (thay đường dẫn đến file best.pt của bạn)
model = YOLO('model/new_vehicle_model.pt') 

# 2. In ra từ điển tên class (Đây là bước quan trọng nhất!)
print("--- CẤU TRÚC CLASS CỦA MODEL ---")
print(model.names) 
# Kết quả mong đợi nếu train 1 class: {0: 'ten_class_cua_ban'}

# 3. (Tùy chọn) Thử chạy trên 1 ảnh bất kỳ để xem nó trả về ID gì
# Thay 'test_image.jpg' bằng đường dẫn ảnh thật
results = model('assets/img/test_img.png') 

for result in results:
    # Lấy các class ID mà model phát hiện được
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    print(f"--- CÁC ID PHÁT HIỆN ĐƯỢC TRONG ẢNH ---")
    print(class_ids)
    # Nếu kết quả in ra là [0 0 0] -> Nghĩa là nó nhận diện ra 3 đối tượng, đều là ID 0.