"""
Tool chuyên nghiệp để lấy tọa độ chính xác từ ảnh 4K
Hỗ trợ zoom in/out, pan, crosshair chính xác
"""

import cv2
import numpy as np


class CoordinatePicker4K:
    def __init__(self, image_path):
        # Đọc ảnh gốc
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Không thể đọc ảnh: {image_path}")
        
        self.image_path = image_path
        self.height, self.width = self.original_image.shape[:2]
        
        # Điểm đã chọn (tọa độ gốc)
        self.points = []
        self.point_names = []  # Tên từng điểm
        
        # Zoom và pan
        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 4.0
        
        # Vị trí hiện tại trên ảnh gốc (center)
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        
        # Kích thước cửa sổ hiển thị
        self.display_width = 1280
        self.display_height = 720
        
        # Trạng thái
        self.is_panning = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.show_grid = True
        self.show_crosshair = True
        
        print(f"\n{'='*60}")
        print(f"🖼️  ẢNH GỐC: {self.width}x{self.height} pixels")
        print(f"🖥️  CỬA SỔ: {self.display_width}x{self.display_height} pixels")
        print(f"{'='*60}\n")
        
    def get_visible_region(self):
        """Tính toán vùng hiển thị trên ảnh gốc"""
        # Kích thước vùng nhìn thấy trên ảnh gốc
        view_width = int(self.display_width / self.zoom_level)
        view_height = int(self.display_height / self.zoom_level)
        
        # Tính góc trái trên
        x1 = max(0, self.center_x - view_width // 2)
        y1 = max(0, self.center_y - view_height // 2)
        
        # Tính góc phải dưới
        x2 = min(self.width, x1 + view_width)
        y2 = min(self.height, y1 + view_height)
        
        # Điều chỉnh lại nếu chạm biên
        if x2 - x1 < view_width:
            x1 = max(0, x2 - view_width)
        if y2 - y1 < view_height:
            y1 = max(0, y2 - view_height)
        
        return x1, y1, x2, y2
    
    def screen_to_original_coords(self, screen_x, screen_y):
        """Chuyển đổi tọa độ màn hình sang tọa độ ảnh gốc"""
        x1, y1, x2, y2 = self.get_visible_region()
        
        # Tỷ lệ giữa vùng nhìn và display
        scale_x = (x2 - x1) / self.display_width
        scale_y = (y2 - y1) / self.display_height
        
        # Tọa độ gốc
        orig_x = int(x1 + screen_x * scale_x)
        orig_y = int(y1 + screen_y * scale_y)
        
        return orig_x, orig_y
    
    def original_to_screen_coords(self, orig_x, orig_y):
        """Chuyển đổi tọa độ ảnh gốc sang tọa độ màn hình"""
        x1, y1, x2, y2 = self.get_visible_region()
        
        scale_x = self.display_width / (x2 - x1)
        scale_y = self.display_height / (y2 - y1)
        
        screen_x = int((orig_x - x1) * scale_x)
        screen_y = int((orig_y - y1) * scale_y)
        
        return screen_x, screen_y
    
    def render_display(self):
        """Vẽ khung hình hiển thị"""
        x1, y1, x2, y2 = self.get_visible_region()
        
        # Crop vùng hiển thị
        roi = self.original_image[y1:y2, x1:x2].copy()
        
        # Resize về kích thước display
        display_image = cv2.resize(roi, (self.display_width, self.display_height), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Vẽ lưới
        if self.show_grid:
            self.draw_grid(display_image)
        
        # Vẽ các điểm đã chọn
        for i, (px, py) in enumerate(self.points):
            screen_x, screen_y = self.original_to_screen_coords(px, py)
            
            # Chỉ vẽ điểm nằm trong vùng hiển thị
            if 0 <= screen_x < self.display_width and 0 <= screen_y < self.display_height:
                cv2.circle(display_image, (screen_x, screen_y), 6, (0, 255, 0), -1)
                cv2.circle(display_image, (screen_x, screen_y), 8, (255, 255, 255), 2)
                
                # Vẽ tên điểm
                label = self.point_names[i] if i < len(self.point_names) else f"P{i+1}"
                cv2.putText(display_image, label, (screen_x + 12, screen_y - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Vẽ đường nối
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                x1, y1 = self.original_to_screen_coords(*self.points[i])
                x2, y2 = self.original_to_screen_coords(*self.points[i+1])
                
                # Vẽ nếu cả 2 điểm trong vùng nhìn
                cv2.line(display_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vẽ thông tin
        self.draw_info(display_image)
        
        return display_image
    
    def draw_grid(self, image):
        """Vẽ lưới tham chiếu"""
        grid_spacing = 100  # Khoảng cách lưới trên display
        
        # Đường dọc
        for x in range(0, self.display_width, grid_spacing):
            cv2.line(image, (x, 0), (x, self.display_height), (80, 80, 80), 1)
        
        # Đường ngang
        for y in range(0, self.display_height, grid_spacing):
            cv2.line(image, (0, y), (self.display_width, y), (80, 80, 80), 1)
    
    def draw_info(self, image):
        """Vẽ thông tin hướng dẫn"""
        info_lines = [
            f"Zoom: {self.zoom_level:.2f}x | Center: ({self.center_x}, {self.center_y})",
            f"Points: {len(self.points)} | Resolution: {self.width}x{self.height}",
            "Controls: Scroll=Zoom | Click=Add | Right=Delete | Drag=Pan | Q=Save | R=Reset | G=Grid | ESC=Exit"
        ]
        
        y_offset = 25
        for line in info_lines:
            cv2.rectangle(image, (5, y_offset - 18), (self.display_width - 5, y_offset + 5), 
                         (0, 0, 0), -1)
            cv2.putText(image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1, cv2.LINE_AA)
            y_offset += 25
    
    def mouse_callback(self, event, x, y, flags, param):
        """Xử lý sự kiện chuột"""
        
        # Scroll để zoom
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # Scroll up
                self.zoom_level = min(self.max_zoom, self.zoom_level * 1.2)
            else:  # Scroll down
                self.zoom_level = max(self.min_zoom, self.zoom_level / 1.2)
            
            print(f"🔍 Zoom: {self.zoom_level:.2f}x")
        
        # Click chuột trái - Thêm điểm
        elif event == cv2.EVENT_LBUTTONDOWN:
            if not self.is_panning:
                orig_x, orig_y = self.screen_to_original_coords(x, y)
                self.points.append((orig_x, orig_y))
                
                # Đặt tên điểm
                point_name = self.get_point_name(len(self.points))
                self.point_names.append(point_name)
                
                print(f"✅ {point_name}: ({orig_x}, {orig_y})")
        
        # Click chuột phải - Xóa điểm cuối
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                removed = self.points.pop()
                removed_name = self.point_names.pop() if self.point_names else f"P{len(self.points)+1}"
                print(f"🗑️  Đã xóa {removed_name}: {removed}")
        
        # Middle button - Pan
        elif event == cv2.EVENT_MBUTTONDOWN:
            self.is_panning = True
            self.last_mouse_x = x
            self.last_mouse_y = y
        
        elif event == cv2.EVENT_MBUTTONUP:
            self.is_panning = False
        
        # Mouse move - Pan nếu đang giữ middle button
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_panning:
                dx = x - self.last_mouse_x
                dy = y - self.last_mouse_y
                
                # Chuyển đổi delta sang tọa độ gốc
                x1, y1, x2, y2 = self.get_visible_region()
                scale = (x2 - x1) / self.display_width
                
                self.center_x = int(self.center_x - dx * scale)
                self.center_y = int(self.center_y - dy * scale)
                
                # Giới hạn
                self.center_x = max(0, min(self.width, self.center_x))
                self.center_y = max(0, min(self.height, self.center_y))
                
                self.last_mouse_x = x
                self.last_mouse_y = y
    
    def get_point_name(self, index):
        """Đặt tên cho điểm dựa vào số thứ tự"""
        if index == 1:
            return "LINE_START"
        elif index == 2:
            return "LINE_END"
        else:
            return f"ZONE_P{index-2}"
    
    def run(self):
        """Chạy tool picker"""
        window_name = "4K Coordinate Picker (Ctrl+Scroll=Zoom)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print("\n" + "="*60)
        print("🎯 HƯỚNG DẪN SỬ DỤNG")
        print("="*60)
        print("🖱️  CHUỘT:")
        print("   - Scroll lên/xuống: Zoom in/out")
        print("   - Click TRÁI: Thêm điểm")
        print("   - Click PHẢI: Xóa điểm cuối")
        print("   - Giữ GIỮA + Kéo: Di chuyển (Pan)")
        print("\n⌨️  PHÍM TẮT:")
        print("   - Q: Lưu và thoát")
        print("   - R: Reset tất cả điểm")
        print("   - G: Bật/tắt lưới")
        print("   - ESC: Thoát không lưu")
        print("="*60 + "\n")
        
        while True:
            display = self.render_display()
            cv2.imshow(window_name, display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Save and quit
                break
            elif key == ord('r'):  # Reset
                self.points = []
                self.point_names = []
                print("🔄 Đã reset tất cả điểm")
            elif key == ord('g'):  # Toggle grid
                self.show_grid = not self.show_grid
            elif key == 27:  # ESC
                self.points = []
                self.point_names = []
                break
        
        cv2.destroyAllWindows()
        return self.points, self.point_names
    
    def save_to_config(self, output_file="coordinates_config.txt"):
        """Lưu tọa độ theo format config.py"""
        if not self.points:
            print("⚠️  Không có điểm để lưu!")
            return
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# " + "="*55 + "\n")
            f.write("# TỌA ĐỘ CHO VEHICLE COUNTING SYSTEM (4K)\n")
            f.write(f"# Ảnh gốc: {self.image_path}\n")
            f.write(f"# Độ phân giải: {self.width}x{self.height}\n")
            f.write(f"# Ngày tạo: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("# " + "="*55 + "\n\n")
            
            # LINE_START và LINE_END
            if len(self.points) >= 2:
                f.write("# Vạch đếm xe (LineZone)\n")
                f.write(f"LINE_START = sv.Point({self.points[0][0]}, {self.points[0][1]})\n")
                f.write(f"LINE_END = sv.Point({self.points[1][0]}, {self.points[1][1]})\n\n")
            
            # LPR Zone (nếu có từ 3 điểm trở lên)
            if len(self.points) >= 3:
                f.write("# Vùng nhận diện biển số (PolygonZone)\n")
                f.write("lpr_zone = np.array([\n")
                for i in range(2, len(self.points)):
                    f.write(f"    [{self.points[i][0]}, {self.points[i][1]}],\n")
                f.write("], dtype=np.int32)\n\n")
            
            # Tất cả điểm (backup)
            f.write("\n# " + "-"*55 + "\n")
            f.write("# Tất cả tọa độ (backup)\n")
            f.write("# " + "-"*55 + "\n")
            for i, (x, y) in enumerate(self.points):
                name = self.point_names[i] if i < len(self.point_names) else f"Point_{i+1}"
                f.write(f"{name} = ({x}, {y})\n")
        
        print(f"\n✅ Đã lưu tọa độ vào: {output_file}")
        print("\n📋 NỘI DUNG FILE:\n")
        with open(output_file, 'r', encoding='utf-8') as f:
            print(f.read())


# ==================== SỬ DỤNG ====================

if __name__ == "__main__":
    
    IMAGE_PATH = "extracted_frames/frame_00030.png"  # ← File ảnh từ bước 1
    
    try:
        picker = CoordinatePicker4K(IMAGE_PATH)
        coordinates, names = picker.run()
        
        if coordinates:
            picker.save_to_config("config_coordinates.txt")
            
            print("\n" + "="*60)
            print("🎉 HOÀN THÀNH!")
            print("="*60)
            print(f"✅ Đã chọn {len(coordinates)} điểm")
            print("📄 File config: config_coordinates.txt")
            print("\n📋 Copy nội dung file này vào config.py của bạn")
            print("="*60)
        else:
            print("\n⚠️  Không có tọa độ nào được chọn")
            
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()