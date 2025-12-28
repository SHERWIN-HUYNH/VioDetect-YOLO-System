import torch

data = torch.load('model/new_vehicle_model.pt', map_location='cpu', weights_only=False)

# 1. Xem các key chính (những ngăn chứa dữ liệu)
print("Các thành phần chính:", data.keys())
# Thường sẽ thấy: ['epoch', 'best_fitness', 'model', 'ema', 'updates', 'optimizer', 'train_args', 'date', 'version']

# 2. Xem danh sách Class được lưu trong model (Metadata)

if 'model' in data:
    try:
        print("Class Names:", data['model'].names)
    except AttributeError:
        # Nếu model là state_dict thuần (ít gặp ở ultralytics)
        print("Model object found but direct names access failed.")
        
# 3. Xem cấu hình train
if 'train_args' in data:
    print(data)
    print("Cấu hình train (imgsz):", data['train_args'].get('imgsz'))