import cv2
import datetime
import os
import re
import numpy as np
from ultralytics import YOLO
import supervision as sv
from paddleocr import PaddleOCR
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ObjectTracking:
    """
    ===== FINAL VERSION (with LPR Zone) =====
    - Uses OCR Confidence for scoring.
    - Uses "Smart Stop" (Confirmation Count + Confirmation Score).
    - Uses "LPR Zone" to prevent wasting attempts on distant vehicles.
    """
    
    # --- CONFIGURATION ---
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]
    VEHICLE_CONF_THRESHOLD = 0.568
    LP_CONF_THRESHOLD = 0.467
    
    OCR_CONFIDENCE_THRESHOLD = 0.15
    MIN_PLATE_LENGTH = 3
    
    # LineZone configuration
    LINE_START = sv.Point(50, 1500)
    LINE_END = sv.Point(3840, 1500)
    
    # ByteTrack configuration
    TRACK_ACTIVATION_THRESHOLD = 0.25
    LOST_TRACK_BUFFER = 60
    MIN_MATCHING_THRESHOLD = 0.8
    FRAME_RATE = 30
    
    # Annotator configuration
    BOX_THICKNESS = 4
    TEXT_THICKNESS = 4
    TEXT_SCALE = 2
    TRACE_THICKNESS = 4
    TRACE_LENGTH = 50
    
    # Performance optimization
    FRAME_SKIP = 3
    LPR_FRAME_INTERVAL = 5
    
    # "Smart Stop" Configuration
    MAX_LPR_ATTEMPTS = 150
    CONFIRMATION_THRESHOLD = 5
    CONFIRMATION_SCORE_THRESHOLD = 5.0 

    
    def __init__(self, vehicle_model_path, lp_model_path, input_source=None, output_source=None, log_file_path="log.txt"):
        """
        input_source: Đường dẫn video HOẶC ảnh
        """
        self.input_source = input_source
        self.output_source = output_source
        self.log_file_path = log_file_path

        self._initialize_models(vehicle_model_path, lp_model_path)
        
        # Chỉ khởi tạo các thành phần video nếu input là video
        self.is_video = False
        if input_source and (input_source.endswith('.mp4') or input_source.endswith('.avi')):
            self.is_video = True
            self._initialize_video_io()
            self._initialize_tracker()
            self._initialize_zone()
            self._initialize_lpr_zone()
            
        self._initialize_annotators()
        
        
        # State tracking (cho video)
        self.plate_texts: Dict[int, str] = {}
        self.plate_candidates: Dict[int, List[Tuple[str, float]]] = {}
        self.lpr_attempts: Dict[int, int] = {}
        self.last_lpr_frame: Dict[int, int] = {}
        self.lp_boxes: Dict[int, np.ndarray] = {}
        self.confirmed_plates: Set[int] = set()
        
        self.vehicle_counts = {
            class_id: {"name": self.CLASS_NAMES_DICT[class_id], "out": 0}
            for class_id in self.VEHICLE_CLASS_IDS
        }
        self.total_lp_detections = 0
        self.total_ocr_successes = 0
        self.total_ocr_attempts = 0

    def _initialize_models(self, vehicle_model_path, lp_model_path):
        """Load YOLO models and PaddleOCR with optimized settings."""
        print("Loading vehicle detection model...")
        self.model = YOLO(vehicle_model_path)
        self.model.fuse()
        
        print("Loading license plate detection model...")
        self.lp_model = YOLO(lp_model_path)
        self.lp_model.fuse()
        
        self.CLASS_NAMES_DICT = self.model.model.names

        print("Initializing PaddleOCR with optimized settings...")
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            use_gpu=True,
            lang='en',
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            rec_batch_num=6,
            show_log=False
        )
        print("Initialization complete.")
    
    def _initialize_video_io(self):
        self.video_info = sv.VideoInfo.from_video_path(self.input_source)
        self.generator = sv.get_video_frames_generator(self.input_source)
        print(f"Video Info: {self.video_info.width}x{self.video_info.height}, "
              f"{self.video_info.fps} FPS, {self.video_info.total_frames} frames")

    def _initialize_tracker(self):
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=self.TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=self.LOST_TRACK_BUFFER,
            minimum_matching_threshold=self.MIN_MATCHING_THRESHOLD,
            frame_rate=self.FRAME_RATE
        )

    def _initialize_zone(self):
        self.line_zone = sv.LineZone(
            start=self.LINE_START,
            end=self.LINE_END,
            triggering_anchors=[sv.Position.BOTTOM_CENTER]
        )

    # ===== HÀM MỚI =====
    def _initialize_lpr_zone(self):
        """Khởi tạo VÙNG mà LPR được phép chạy."""
        print("Initializing LPR Activation Zone...")
        
        # Lấy chiều cao của video (ví dụ: 2160)
        frame_height = self.video_info.height
        # Lấy chiều rộng của video (ví dụ: 3840)
        frame_width = self.video_info.width
        
        # Tính toán điểm giữa theo chiều dọc
        mid_height = frame_height // 2 
        
       
        lpr_roi_polygon = np.array([
            [0, mid_height],             # Top-left (0, 1080)
            [frame_width, mid_height],   # Top-right (3840, 1080)
            [frame_width, frame_height], # Bottom-right (3840, 2160)
            [0, frame_height]            # Bottom-left (0, 2160)
        ])
        # ==================================
        
        self.lpr_zone = sv.PolygonZone(
            polygon=lpr_roi_polygon, 
            frame_resolution_wh=self.video_info.resolution_wh,
            triggering_anchors=[sv.Position.BOTTOM_CENTER]
        )

    def _initialize_annotators(self):
        """Initialize supervision annotators."""
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=self.BOX_THICKNESS)
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=self.TEXT_THICKNESS, 
            text_scale=self.TEXT_SCALE
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=self.TRACE_THICKNESS, 
            trace_length=self.TRACE_LENGTH
        )
        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=self.BOX_THICKNESS,
            text_thickness=self.TEXT_THICKNESS,
            text_scale=self.TEXT_SCALE,
            color=sv.Color.RED,
            display_in_count=False,
            display_out_count=True
        )
        self.lp_box_annotator = sv.BoundingBoxAnnotator(
            thickness=self.BOX_THICKNESS,
            color=sv.Color.GREEN
        )
        self.raw_lp_box_annotator = sv.BoundingBoxAnnotator(
            thickness=2,
            color=sv.Color.YELLOW
        )
        self.lp_label_annotator = sv.LabelAnnotator(
            text_thickness=self.TEXT_THICKNESS, 
            text_scale=self.TEXT_SCALE,
            text_color=sv.Color.WHITE
        )
        
        
        if hasattr(self, 'lpr_zone'):
            self.lpr_zone_annotator = sv.PolygonZoneAnnotator(
                zone=self.lpr_zone,
                color=sv.Color.GREEN,
                thickness=2,
                text_scale=1,
                text_thickness=2,
                text_padding=5
            )
        else:
            self.lpr_zone_annotator = None

    def process_image(self):
        """Xử lý trên 1 ảnh tĩnh."""
        print(f"Processing image: {self.input_source}")
        frame = cv2.imread(self.input_source)
        if frame is None:
            print("Error: Could not read image.")
            return

        # 1. Detect Vehicles
        vehicle_detections = self._detect_vehicles(frame)
        print(f"Detected {len(vehicle_detections)} vehicles.")

        plate_labels = []
        lp_xyxy_list = []
        lp_class_list = []
        final_labels = []

        # 2. Loop qua từng xe để tìm biển số (Logic đơn giản hóa cho ảnh)
        for i, (xyxy, conf, class_id) in enumerate(zip(vehicle_detections.xyxy, vehicle_detections.confidence, vehicle_detections.class_id)):
            vehicle_name = self.CLASS_NAMES_DICT.get(class_id, "Vehicle")
            plate_text = ""
            
            # Crop xe
            x1, y1, x2, y2 = map(int, xyxy)
            vehicle_crop = frame[y1:y2, x1:x2]
            
            if vehicle_crop.size > 0:
                # Detect Plate (YOLO LP)
                lp_results = self.lp_model(vehicle_crop, verbose=False)[0]
                lp_detections = sv.Detections.from_ultralytics(lp_results)
                
                # Nếu tìm thấy biển số
                if len(lp_detections) > 0:
                    best_idx = np.argmax(lp_detections.confidence)
                    lpx1, lpy1, lpx2, lpy2 = map(int, lp_detections.xyxy[best_idx])
                    
                    lp_crop = vehicle_crop[lpy1:lpy2, lpx1:lpx2]
                    
                    # OCR
                    ocr_result = self.ocr.ocr(lp_crop, cls=False)
                    text, conf_ocr = self._clean_ocr_text(ocr_result)
                    
                    if text and len(text) >= self.MIN_PLATE_LENGTH:
                        plate_text = text
                        # Lưu tọa độ tuyệt đối của biển số để vẽ
                        abs_lp_box = [lpx1 + x1, lpy1 + y1, lpx2 + x1, lpy2 + y1]
                        lp_xyxy_list.append(abs_lp_box)
                        plate_labels.append(text)
                        lp_class_list.append(0)
                        print(f"  -> Vehicle {i}: Found Plate '{text}'")

            # Tạo nhãn cho xe
            label = f"{vehicle_name.upper()}"
            if plate_text:
                label += f" | {plate_text}"
            final_labels.append(label)

        # 3. Vẽ lên ảnh
        # Vẽ xe
        frame = self.box_annotator.annotate(scene=frame, detections=vehicle_detections)
        frame = self.label_annotator.annotate(scene=frame, detections=vehicle_detections, labels=final_labels)

        # Vẽ biển số (nếu có)
        if lp_xyxy_list:
            lp_detections = sv.Detections(xyxy=np.array(lp_xyxy_list), class_id=np.array(lp_class_list))
            frame = self.lp_box_annotator.annotate(scene=frame, detections=lp_detections)
            frame = self.lp_label_annotator.annotate(scene=frame, detections=lp_detections, labels=plate_labels)

        # Vẽ thống kê đơn giản
        self._draw_image_stats(frame, len(vehicle_detections), len(lp_xyxy_list))

        # 4. Lưu ảnh
        cv2.imwrite(self.output_source, frame)
        print(f"Saved result to: {self.output_source}")

    def _draw_image_stats(self, frame, vehicle_count, plate_count):
        """Vẽ bảng thống kê cho ảnh tĩnh."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (400, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, "ANALYSIS RESULT", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Plates: {plate_count}", (40, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame
    
    def _detect_vehicles(self, frame: np.ndarray) -> sv.Detections:
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence > self.VEHICLE_CONF_THRESHOLD]
        detections = detections[np.isin(detections.class_id, self.VEHICLE_CLASS_IDS)]
        return detections

    def _update_counts(self, tracked_detections: sv.Detections):
        if tracked_detections.tracker_id is None:
            return
        _, crossed_out = self.line_zone.trigger(tracked_detections)
        for i, (tracker_id, class_id) in enumerate(
            zip(tracked_detections.tracker_id, tracked_detections.class_id)
        ):
            if class_id in self.vehicle_counts:
                if crossed_out[i]:
                    self.vehicle_counts[class_id]["out"] += 1
                    print(f"  ↑ Vehicle #{tracker_id} ({self.CLASS_NAMES_DICT[class_id]}) crossed OUT")

    def _clean_ocr_text(self, ocr_result) -> Tuple[str, float]:
        """
        Trả về (text, avg_confidence)
        """
        if ocr_result is None or not ocr_result:
            return "", 0.0
        
        try:
            text_parts = []
            conf_parts = []
            
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                if isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
                    for item in ocr_result[0]:
                        if len(item) >= 2 and isinstance(item[1], (tuple, list)) and len(item[1]) >= 2:
                            text, confidence = item[1][0], item[1][1]
                            if text:
                                text_parts.append(str(text))
                                conf_parts.append(confidence)
                
            if not text_parts:
                return "", 0.0
            
            full_text = ''.join(text_parts)
            cleaned = re.sub(r'[^A-Z0-9]', '', full_text.upper())
            
            corrections = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'}
            result = ""
            for i, char in enumerate(cleaned):
                if char in corrections:
                    prev_is_digit = (i > 0 and cleaned[i-1].isdigit())
                    next_is_digit = (i < len(cleaned)-1 and cleaned[i+1].isdigit())
                    if prev_is_digit and next_is_digit:
                        result += corrections[char]
                    else:
                        result += char
                else:
                    result += char
            
            avg_conf = sum(conf_parts) / len(conf_parts) if conf_parts else 0.0
            
            if avg_conf < self.OCR_CONFIDENCE_THRESHOLD:
                # print(f"    [OCR] ❌ Discarded: '{result}' (Avg Conf: {avg_conf:.3f} < {self.OCR_CONFIDENCE_THRESHOLD})")
                return "", 0.0

            # print(f"    [OCR] ✅ Final: '{result}' (Avg Conf: {avg_conf:.3f})")
            return cleaned, avg_conf
            
        except Exception as e:
            print(f"    [OCR ERROR] {e}")
            return "", 0.0

    def _should_process_lpr(self, tracker_id: int, frame_index: int) -> bool:
        """
        Logic "Smart Stop"
        """
        
        if tracker_id in self.confirmed_plates:
            return False
            
        if self.lpr_attempts.get(tracker_id, 0) >= self.MAX_LPR_ATTEMPTS:
            if self.lpr_attempts.get(tracker_id, 0) == self.MAX_LPR_ATTEMPTS:
                 print(f"  [Vehicle #{tracker_id}] Skipped - Max attempts ({self.MAX_LPR_ATTEMPTS}) reached INSIDE ZONE")
            return False
            
        last_frame = self.last_lpr_frame.get(tracker_id, -999)
        if frame_index - last_frame < self.LPR_FRAME_INTERVAL:
            return False
            
        return True

    def _select_best_plate(self, tracker_id: int) -> Optional[str]:
        """
        Logic "Xác nhận Thông minh"
        """
        if tracker_id not in self.plate_candidates:
            return None
        
        candidates = self.plate_candidates[tracker_id]
        if not candidates:
            return None
        
        best_text = None
        best_score = 0
        
        for text, conf in candidates:
            score = len(text) * conf 
            if score > best_score:
                best_score = score
                best_text = text
        
        # Logic xác nhận: Phải đạt cả SỐ LẦN và CHẤT LƯỢNG
        if best_text and tracker_id not in self.confirmed_plates:
            vote_count = Counter([txt for txt, conf in candidates])
            count = vote_count.get(best_text, 0)
            
            if count >= self.CONFIRMATION_THRESHOLD and best_score >= self.CONFIRMATION_SCORE_THRESHOLD:
                self.confirmed_plates.add(tracker_id)
                print(f"  [VOTING] ✅✅ CONFIRMED: Vehicle #{tracker_id} is '{best_text}' (Score: {best_score:.2f}, Votes: {count})")
        
        return best_text

    # ===== HÀM ĐƯỢC CẬP NHẬT QUAN TRỌNG =====
    def _process_lpr(self, frame: np.ndarray, tracked_detections: sv.Detections, frame_index: int):
        if tracked_detections.tracker_id is None:
            return
        
        # ===== THÊM MỚI: Lấy các xe ở trong LPR Zone =====
        try:
            # Lấy mask (True/False) cho các xe ở trong LPR Zone
            mask = self.lpr_zone.trigger(detections=tracked_detections)
            # Lọc ra chỉ các xe trong zone
            detections_in_zone = tracked_detections[mask]
        except Exception as e:
            detections_in_zone = sv.Detections.empty()
        # ===============================================

        # ===== THAY ĐỔI: Chỉ lặp qua các xe TRONG VÙNG LPR =====
        for idx, (xyxy, conf, class_id, tracker_id) in enumerate(zip(
            detections_in_zone.xyxy,
            detections_in_zone.confidence,
            detections_in_zone.class_id,
            detections_in_zone.tracker_id
        )):
            
            # `lpr_attempts` sẽ chỉ tăng NẾU xe ở trong vùng.
            if not self._should_process_lpr(tracker_id, frame_index):
                continue
            
            try:
                # Cập nhật trạng thái
                self.last_lpr_frame[tracker_id] = frame_index
                self.lpr_attempts[tracker_id] = self.lpr_attempts.get(tracker_id, 0) + 1
                attempt_num = self.lpr_attempts[tracker_id]
                
                print(f"\n  [Vehicle #{tracker_id}] Attempt {attempt_num}/{self.MAX_LPR_ATTEMPTS} (Inside Zone)")
                
                # --- Bắt đầu luồng LPR ---
                
                # 1. Crop vehicle
                x1, y1, x2, y2 = map(int, xyxy)
                pad = 10
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
                
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size == 0 or vehicle_crop.shape[0] < 20 or vehicle_crop.shape[1] < 20:
                    continue
                
                # 2. Detect LP (YOLO-LP)
                lp_results = self.lp_model(vehicle_crop, verbose=False)[0]
                lp_detections = sv.Detections.from_ultralytics(lp_results)
                lp_detections = lp_detections[lp_detections.confidence > self.LP_CONF_THRESHOLD]
                
                if len(lp_detections) == 0:
                    continue
                
                # Thêm vào annotator (raw detections)
                abs_xyxy = lp_detections.xyxy.copy()
                abs_xyxy[:, 0] += x1; abs_xyxy[:, 1] += y1; abs_xyxy[:, 2] += x1; abs_xyxy[:, 3] += y1
                abs_lp_detections = sv.Detections(xyxy=abs_xyxy, confidence=lp_detections.confidence, class_id=lp_detections.class_id)
                self.current_frame_lp_detections = sv.Detections.merge([self.current_frame_lp_detections, abs_lp_detections])
                
                # 3. Get best LP
                best_idx = np.argmax(lp_detections.confidence)
                relative_lp_box = lp_detections.xyxy[best_idx]
                
                self.total_lp_detections += 1
                lx1, ly1, lx2, ly2 = map(int, relative_lp_box)
                lp_crop = vehicle_crop[ly1:ly2, lx1:lx2]
                
                if lp_crop.size == 0 or lp_crop.shape[0] < 10 or lp_crop.shape[1] < 20:
                    continue
                
                # 4. Run OCR (PaddleOCR)
                self.total_ocr_attempts += 1
                try:
                    ocr_result = self.ocr.ocr(lp_crop, cls=False)
                except Exception as ocr_e:
                    print(f"  [Vehicle #{tracker_id}] OCR Exception: {ocr_e}")
                    ocr_result = None
                
                # 5. Clean text và lấy (text, ocr_conf)
                plate_text, ocr_conf = self._clean_ocr_text(ocr_result) if ocr_result else ("", 0.0)
                
                # 6. Save candidate
                if plate_text and len(plate_text) >= self.MIN_PLATE_LENGTH and ocr_conf > 0:
                    self.total_ocr_successes += 1
                    
                    if tracker_id not in self.plate_candidates:
                        self.plate_candidates[tracker_id] = []
                    
                    self.plate_candidates[tracker_id].append((plate_text, ocr_conf))
                    print(f"  [Vehicle #{tracker_id}] ✅ Candidate saved: '{plate_text}' (OCR conf={ocr_conf:.3f})")
                    
                    # 7. Select best plate
                    best_plate = self._select_best_plate(tracker_id)
                    if best_plate:
                        self.plate_texts[tracker_id] = best_plate
                        
                        abs_lp_box = [relative_lp_box[0] + x1, relative_lp_box[1] + y1, relative_lp_box[2] + x1, relative_lp_box[3] + y1]
                        self.lp_boxes[tracker_id] = np.array(abs_lp_box)
                else:
                    if not (plate_text and len(plate_text) >= self.MIN_PLATE_LENGTH):
                        # print(f"  [Vehicle #{tracker_id}] ❌ Invalid: '{plate_text}' (len={len(plate_text)})")
                        pass
            
            except Exception as e:
                print(f"  [Vehicle #{tracker_id}] Error: {e}")

    def _annotate_frame(self, frame: np.ndarray, tracked_detections: sv.Detections, 
                       frame_index: int) -> np.ndarray:
        labels = []
        if tracked_detections.tracker_id is not None:
            for confidence, class_id, tracker_id in zip(
                tracked_detections.confidence, 
                tracked_detections.class_id, 
                tracked_detections.tracker_id
            ):
                vehicle_name = self.CLASS_NAMES_DICT.get(class_id, "Unknown")
                plate_text = self.plate_texts.get(tracker_id)
                
                label = f"#{tracker_id}"
                if plate_text:
                    label += f" | {plate_text}"
                else:
                    label += f" {vehicle_name.upper()}"
                
                if tracker_id in self.confirmed_plates:
                    label += " [OK]"
                

                labels.append(label)

            frame = self.trace_annotator.annotate(scene=frame, detections=tracked_detections)
            frame = self.box_annotator.annotate(scene=frame, detections=tracked_detections)
            frame = self.label_annotator.annotate(
                scene=frame, detections=tracked_detections, labels=labels
            )
            frame = self.raw_lp_box_annotator.annotate(
                scene=frame, detections=self.current_frame_lp_detections
            )
        
        frame = self.draw_statistics(frame, frame_index)
        
        # Vẽ các vùng
        frame = self.line_zone_annotator.annotate(frame, line_counter=self.line_zone)
        # ===== THÊM MỚI: Vẽ LPR Zone =====
        frame = self.lpr_zone_annotator.annotate(frame, label="LPR Activation Zone")
        # ================================

        # Vẽ biển số đã xác nhận
        lp_xyxy_list, lp_label_list, lp_class_id_list = [], [], []
        for tracker_id, lp_box in self.lp_boxes.items():
            if tracker_id in tracked_detections.tracker_id:
                lp_xyxy_list.append(lp_box)
                lp_label_list.append(self.plate_texts.get(tracker_id, ""))
                lp_class_id_list.append(0)

        if lp_xyxy_list:
            lp_detections = sv.Detections(
                xyxy=np.array(lp_xyxy_list),
                class_id=np.array(lp_class_id_list)
            )
            frame = self.lp_box_annotator.annotate(scene=frame, detections=lp_detections)
            frame = self.lp_label_annotator.annotate(
                scene=frame, detections=lp_detections, labels=lp_label_list
            )
        
        return frame

    def callback(self, frame: np.ndarray, index: int) -> np.ndarray:
        self.current_frame_lp_detections = sv.Detections.empty()
        detections = self._detect_vehicles(frame)
        tracked_detections = self.byte_tracker.update_with_detections(detections)
        self._update_counts(tracked_detections)
        
        # Truyền cả detections và tracked_detections
        # Chúng ta cần 'tracked_detections' để lọc LPR zone
        self._process_lpr(frame, tracked_detections, index) 
        
        annotated_frame = self._annotate_frame(frame.copy(), tracked_detections, index)
        return annotated_frame

    def process(self):
        import time  # Import thư viện đo thời gian
        
        frame_count, processed_count = 0, 0
        try:
            # Lưu ý: Sửa self.output_video_path -> self.output_source để khớp với __init__
            with sv.VideoSink(target_path=self.output_source, video_info=self.video_info) as sink:
                
                # [BẮT ĐẦU] Khởi tạo thời gian
                prev_time = time.time()

                for index, frame in enumerate(self.generator):
                    frame_count += 1
                    
                    # Logic Frame Skip: Chỉ xử lý frame chia hết cho FRAME_SKIP
                    if index % self.FRAME_SKIP != 0:
                        sink.write_frame(frame=frame)
                        continue
                    
                    # Log tiến độ mỗi 30 frame
                    if index % 30 == 0:
                        print(f"Processing frame {index}/{self.video_info.total_frames} "
                              f"({100*index/self.video_info.total_frames:.1f}%)")
                    
                    # --- XỬ LÝ CHÍNH (Detect + Track + OCR) ---
                    result_frame = self.callback(frame, index)
                    
                    # [FPS] Tính toán FPS thực tế
                    curr_time = time.time()
                    delta_time = curr_time - prev_time
                    prev_time = curr_time  # Cập nhật mốc thời gian
                    
                    fps = 1 / delta_time if delta_time > 0 else 0
                    
                    # [FPS] In ra console
                    print(f"Frame {index} - Real-time FPS: {fps:.2f}")

                    # [FPS] Vẽ lên hình (Góc trên bên trái, Màu đỏ)
                    cv2.putText(
                        result_frame, 
                        f"FPS: {fps:.2f}", 
                        (20, 50),                 # Tọa độ (x, y)
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2,                      # Font scale
                        (0, 0, 255),              # Màu đỏ (BGR)
                        3                         # Độ dày nét
                    )

                    # Ghi frame kết quả vào video đầu ra
                    sink.write_frame(frame=result_frame)
                    processed_count += 1

                    display_frame = cv2.resize(result_frame, (1280, 720))
                    cv2.imshow("Real-time Processing", display_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\n[INFO] Dừng chạy bởi người dùng (Pressed 'q').")
                        break
                        
        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"\nError inside process loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"\n{'='*60}")
            print(f"Processed: {processed_count}/{frame_count} frames")
            print(f"LP detections: {self.total_lp_detections}")
            print(f"OCR attempts (Inside Zone): {self.total_ocr_attempts}")
            print(f"OCR successes: {self.total_ocr_successes}")
            print(f"Unique plates: {len(self.plate_texts)}")
            print(f"Confirmed plates: {len(self.confirmed_plates)}")
            print(f"{'='*60}")
            self.save_log()

    def draw_statistics(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        overlay = frame.copy()
        height, width = frame.shape[:2]
        PANEL_WIDTH, PANEL_HEIGHT = 600, 500
        PANEL_MARGIN, PANEL_X = 20, width - 620
        PANEL_Y, ALPHA = 20, 0.7
        
        cv2.rectangle(overlay, (PANEL_X, PANEL_Y), 
                     (PANEL_X + PANEL_WIDTH, PANEL_Y + PANEL_HEIGHT), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)

        FONT = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "VEHICLE STATISTICS", (PANEL_X + 20, PANEL_Y + 60), 
                   FONT, 1.3, (255, 255, 255), 3)
        cv2.line(frame, (PANEL_X + 20, PANEL_Y + 80), 
                (PANEL_X + PANEL_WIDTH - 20, PANEL_Y + 80), (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_index}", (PANEL_X + 20, PANEL_Y + 120), 
                   FONT, 0.8, (200, 200, 200), 2)

        total_out = sum(v["out"] for v in self.vehicle_counts.values())
        cv2.putText(frame, f"TOTAL OUT: {total_out}", (PANEL_X + 20, PANEL_Y + 180), 
                   FONT, 1.1, (0, 100, 255), 3)

        y_offset = PANEL_Y + 260
        colors = {2: (255, 200, 0), 3: (255, 100, 255), 5: (0, 255, 255), 7: (100, 255, 100)}
        for class_id in self.VEHICLE_CLASS_IDS:
            vehicle = self.vehicle_counts[class_id]
            cv2.putText(frame, vehicle["name"].upper(), (PANEL_X + 20, y_offset), FONT, 1, 
                       colors.get(class_id, (255, 255, 255)), 3)
            cv2.putText(frame, f"OUT: {vehicle['out']}", (PANEL_X + 200, y_offset), 
                       FONT, 1, (0, 100, 255), 3)
            y_offset += 50
        return frame

    def save_log(self):
        """Save final counts and detected plates to log file."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] - Processing Complete\n"
            log_message += f"Input Video: {self.input_source}\n"
            log_message += f"Output Video: {self.output_source}\n"
            log_message += f"Total Frames: {self.video_info.total_frames}\n"
            
            log_message += "\n=== FINAL COUNTS ===\n"
            total_out = 0
            for class_id, data in self.vehicle_counts.items():
                name = data['name'].title()
                out_count = data['out']
                log_message += f"{name}: OUT={out_count}\n"
                total_out += out_count
            log_message += "--------------------\n"
            log_message += f"TOTAL OUT: {total_out}\n"
            
            log_message += "\n=== LPR STATISTICS ===\n"
            log_message += f"Total YOLO-LP Detections (Events): {self.total_lp_detections}\n"
            log_message += f"Total OCR Attempts (Inside Zone): {self.total_ocr_attempts}\n"
            log_message += f"Total Successful OCR Reads: {self.total_ocr_successes}\n"
            log_message += f"Total Plates Confirmed: {len(self.confirmed_plates)}\n"

            log_message += "\n=== DETECTED PLATES ===\n"
            if self.plate_texts:
                for tracker_id, plate in sorted(self.plate_texts.items()):
                    log_message += f"Vehicle #{tracker_id}: {plate}\n"
                log_message += f"\nTotal plates detected: {len(self.plate_texts)}\n"
            else:
                log_message += "No plates detected.\n"

            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(log_message)
            print(f"\n✓ Log saved successfully to {self.log_file_path}")

        except Exception as e:
            print(f"Error saving log file: {e}")


# --- ENTRY POINT ---
if __name__ == "__main__":
    
    VEHICLE_MODEL_PATH = "model/vehicle_detector.pt"
    LP_MODEL_PATH = "model/lpr_detector.pt"
    
    # === CHỌN CHẾ ĐỘ Ở ĐÂY ===
    MODE = "VIDEO" # "VIDEO" hoặc "IMAGE"
    if MODE == "VIDEO":
        INPUT = "assets/video/license-counting-video.mp4"
        OUTPUT = "assets/video/result.mp4"
    else:
        INPUT = "assets/img/test_img.png"  # Đường dẫn ảnh đầu vào
        OUTPUT = "assets/img/result.png"   # Đường dẫn ảnh đầu ra

    try:
        tracker = ObjectTracking(
            vehicle_model_path=VEHICLE_MODEL_PATH,
            lp_model_path=LP_MODEL_PATH,
            input_source=INPUT,
            output_source=OUTPUT
        )
        
        if MODE == "VIDEO":
            tracker.process() # Chạy chế độ video cũ
        else:
            tracker.process_image() # Chạy chế độ ảnh mới
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()