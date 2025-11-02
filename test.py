import cv2
import datetime  
import numpy as np
from ultralytics import YOLO
import supervision as sv  



class ObjectTracking:
    def __init__(self, input_video_path, output_video_path, log_file_path) -> None:
        self.model = YOLO("model/vehicle_detector.pt")
        self.model.fuse()

        # dict maping class_id to class_name
        self.CLASS_NAMES_DICT = self.model.model.names
        # class_ids of interest - car, motocycle, bus and truck
        self.CLASS_ID = [2, 3, 5, 7]

        self.LINE_START = sv.Point(50, 1500)
        self.LINE_END = sv.Point(3840, 1500)

        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.log_file_path = log_file_path 

        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=60,  
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        self.video_info = sv.VideoInfo.from_video_path(self.input_video_path)
        self.generator = sv.get_video_frames_generator(self.input_video_path)

        self.line_zone = sv.LineZone(
            start=self.LINE_START,
            end=self.LINE_END,
            triggering_anchors=[sv.Position.BOTTOM_CENTER]
        )

        
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=4)
        self.label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
        self.trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)
        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=4,
            text_thickness=4,
            text_scale=2
        )

        self.counted_ids = set()

        self.vehicle_counts = {
            2: {"name": "car", "in": 0, "out": 0},
            3: {"name": "motorcycle", "in": 0, "out": 0},
            5: {"name": "bus", "in": 0, "out": 0},
            7: {"name": "truck", "in": 0, "out": 0}
        }

    def callback(self, frame, index):
        # model prediction on single frame and conversion to supervision Detections
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        detections = detections[detections.confidence > 0.3]
        detections = detections[np.isin(detections.class_id, self.CLASS_ID)]

        detections = self.byte_tracker.update_with_detections(detections)

        annotated_frame = frame.copy()

        crossed_in, crossed_out = self.line_zone.trigger(detections)

        # Cập nhật counter cho từng loại xe
        if detections.tracker_id is not None:
            for i, (tracker_id, class_id) in enumerate(zip(detections.tracker_id, detections.class_id)):
                # Đảm bảo class_id nằm trong danh sách theo dõi
                if class_id in self.vehicle_counts:
                    if crossed_in[i]:
                        self.vehicle_counts[class_id]["in"] += 1
                        print(f"✓ Frame {index}: {self.vehicle_counts[class_id]['name'].upper()} #{tracker_id} crossed IN")
                    elif crossed_out[i]:
                        self.vehicle_counts[class_id]["out"] += 1
                        print(f"✓ Frame {index}: {self.vehicle_counts[class_id]['name'].upper()} #{tracker_id} crossed OUT")

        # Debug: In ra thông tin về các xe vừa qua line
        if len(crossed_in) > 0 or len(crossed_out) > 0:
            print(f"Frame {index}: Total IN={len(crossed_in)}, Total OUT={len(crossed_out)}")

        if detections.tracker_id is not None and len(detections.tracker_id) > 0:
            labels = []
            for confidence, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id):
                # Highlight xe vừa qua line bằng màu khác
                vehicle_name = self.model.model.names[class_id]
                label = f"#{tracker_id} {vehicle_name} {confidence:0.2f}"
                labels.append(label)

            # annotate with traces
            annotated_frame = self.trace_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

            # annotate with bounding boxes
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame,
                detections=detections
            )

            # annotate with labels
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=detections,
                labels=labels
            )

        # VẼ BẢNG THỐNG KÊ LÊN VIDEO
        annotated_frame = self.draw_statistics(annotated_frame, index)

        # return frame with box and line annotated result
        return self.line_zone_annotator.annotate(annotated_frame, line_counter=self.line_zone)

    def draw_statistics(self, frame, frame_index):
        """Vẽ bảng thống kê lên video"""
        # import cv2

        overlay = frame.copy()
        height, width = frame.shape[:2]

        # Kích thước và vị trí bảng thống kê
        panel_width = 600
        panel_height = 500
        panel_x = width - panel_width - 20
        panel_y = 20

        # Vẽ background với độ trong suốt
        cv2.rectangle(overlay,
                      (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (0, 0, 0),
                      -1)

        # Blend overlay với frame gốc
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Tiêu đề
        cv2.putText(frame, "VEHICLE STATISTICS",
                    (panel_x + 20, panel_y + 60), # Tăng Y
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3) # Tăng scale & thickness

        cv2.line(frame,
                 (panel_x + 20, panel_y + 80), # Tăng Y
                 (panel_x + panel_width - 20, panel_y + 80),
                 (255, 255, 255), 2)

        # Frame info
        cv2.putText(frame, f"Frame: {frame_index}",
                    (panel_x + 20, panel_y + 120), # Tăng Y
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2) # Tăng scale & thickness

        # Tổng số xe
        total_in = sum(v["in"] for v in self.vehicle_counts.values())
        total_out = sum(v["out"] for v in self.vehicle_counts.values())

        cv2.putText(frame, f"TOTAL IN: {total_in}",
                    (panel_x + 20, panel_y + 180), # Tăng Y
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3) # Tăng scale & thickness

        cv2.putText(frame, f"TOTAL OUT: {total_out}",
                    (panel_x + 350, panel_y + 180), # Tăng Y và X
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 100, 255), 3) # Tăng scale & thickness

        # Chi tiết từng loại xe
        y_offset = 260 # Tăng Y bắt đầu

        colors = {
            2: (255, 200, 0),    # Car - Cyan
            3: (255, 100, 255),  # Motorcycle - Magenta
            5: (0, 255, 255),    # Bus - Yellow
            7: (100, 255, 100)   # Truck - Light Green
        }

        for class_id in [2, 3, 5, 7]:
            vehicle = self.vehicle_counts[class_id]
            color = colors[class_id]

            # Tên loại xe
            name = vehicle["name"].upper()
            cv2.putText(frame, name,
                        (panel_x + 20, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

            # IN count
            cv2.putText(frame, f"IN: {vehicle['in']}",
                        (panel_x + 200, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # OUT count
            cv2.putText(frame, f"OUT: {vehicle['out']}",
                        (panel_x + 310, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 3)

            y_offset += 50

        # Vẽ legend cho line
        legend_y = panel_y + panel_height - 40
        cv2.putText(frame, "GREEN LINE: Counting Zone",
                    (panel_x + 20, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame

    # 2. Thêm hàm 'save_log' mới
    def save_log(self):
        """Lưu kết quả đếm cuối cùng vào file log."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] - Processing Complete\n"
            log_message += f"Input Video: {self.input_video_path}\n"
            log_message += f"Output Video: {self.output_video_path}\n"
            log_message += "\n=== FINAL COUNTS ===\n"

            total_in = 0
            total_out = 0

            for class_id, data in self.vehicle_counts.items():
                name = data['name'].title() # Viết hoa chữ cái đầu
                in_count = data['in']
                out_count = data['out']
                log_message += f"{name}: IN={in_count}, OUT={out_count}\n"
                total_in += in_count
                total_out += out_count
            
            log_message += "--------------------\n"
            log_message += f"TOTAL IN: {total_in}\n"
            log_message += f"TOTAL OUT: {total_out}\n"

            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(log_message)
            print(f"Log saved successfully to {self.log_file_path}")

        except Exception as e:
            print(f"Error saving log file: {e}")

    # 3. Cập nhật hàm 'process'
    def process(self):
        # Thêm 'try...finally' để đảm bảo log được lưu và cửa sổ được đóng
        try:
            with sv.VideoSink(target_path=self.output_video_path, video_info=self.video_info) as sink:
                for index, frame in enumerate(
                    sv.get_video_frames_generator(source_path=self.input_video_path)
                ):
                    result_frame = self.callback(frame, index)
                    sink.write_frame(frame=result_frame)

                    # --- BỔ SUNG: HIỂN THỊ VIDEO ---
                    h, w, _ = result_frame.shape
                    display_width = w // 3 
                    display_height = h // 3
                    display_frame = cv2.resize(result_frame, (display_width, display_height))

                    # Hiển thị
                    cv2.imshow("Vehicle Counting (Press 'q' to quit)", display_frame)

                    # Nhấn 'q' để thoát
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("Đã nhấn 'q', dừng xử lý...")
                        break

        finally:
            cv2.destroyAllWindows()  
            print("\nProcessing finished. Saving log...")
            # Gọi hàm lưu log
            self.save_log()
            # --- KẾT THÚC BỔ SUNG ---


# # --- CÁCH SỬ DỤNG MỚI ---
# if __name__ == "__main__":
#     INPUT_PATH = "assets/video/vehicle-counting.mp4"
#     OUTPUT_PATH = "assets/video/vehicle-counting-result.mp4"
#     LOG_PATH = "vehicle_count_log.txt"  # Đặt tên file log

#     # Khởi tạo và chạy (thêm LOG_PATH)
#     obj = ObjectTracking(INPUT_PATH, OUTPUT_PATH, LOG_PATH)
#     obj.process()