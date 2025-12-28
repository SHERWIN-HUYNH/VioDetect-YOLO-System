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
import threading
import queue
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ObjectTracking:
    """
    OPTIMIZED VERSION - Major FPS improvements:
    1. Asynchronous OCR processing
    2. Batch LP detection
    3. Skip frame writing for skipped frames
    4. Optional display toggle
    5. Proper FPS calculation
    """
    
    # --- CONFIGURATION ---
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]
    VEHICLE_CONF_THRESHOLD = 0.568
    LP_CONF_THRESHOLD = 0.467
    
    OCR_CONFIDENCE_THRESHOLD = 0.15
    MIN_PLATE_LENGTH = 3
    
    LINE_START = sv.Point(50, 1500)
    LINE_END = sv.Point(3840, 1500)
    
    TRACK_ACTIVATION_THRESHOLD = 0.25
    LOST_TRACK_BUFFER = 60
    MIN_MATCHING_THRESHOLD = 0.8
    FRAME_RATE = 30
    
    BOX_THICKNESS = 4
    TEXT_THICKNESS = 4
    TEXT_SCALE = 2
    TRACE_THICKNESS = 4
    TRACE_LENGTH = 50
    
    #  OPTIMIZED SETTINGS
    FRAME_SKIP = 3
    LPR_FRAME_INTERVAL = 10  # Tăng từ 5 lên 10
    ENABLE_DISPLAY = False   # TẮT hiển thị real-time
    
    MAX_LPR_ATTEMPTS = 150
    CONFIRMATION_THRESHOLD = 5
    CONFIRMATION_SCORE_THRESHOLD = 5.0

    
    def __init__(self, vehicle_model_path, lp_model_path, input_source=None, 
                 output_source=None, log_file_path="log.txt"):
        self.input_source = input_source
        self.output_source = output_source
        self.log_file_path = log_file_path

        self._initialize_models(vehicle_model_path, lp_model_path)
        
        self.is_video = False
        if input_source and (input_source.endswith('.mp4') or input_source.endswith('.avi')):
            self.is_video = True
            self._initialize_video_io()
            self._initialize_tracker()
            self._initialize_zone()
            self._initialize_lpr_zone()
            
        self._initialize_annotators()
        
        #  ASYNC OCR SETUP
        self.ocr_queue = queue.Queue(maxsize=20)
        self.ocr_result_queue = queue.Queue()
        self.ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self.ocr_thread_active = True
        self.ocr_thread.start()
        
        # State tracking
        self.plate_texts: Dict[int, str] = {}
        self.plate_candidates: Dict[int, List[Tuple[str, float]]] = {}
        self.lpr_attempts: Dict[int, int] = {}
        self.last_lpr_frame: Dict[int, int] = {}
        self.lp_boxes: Dict[int, np.ndarray] = {}
        self.confirmed_plates: Set[int] = set()
        
        self.vehicle_counts = {
            class_id: {"name": self.CLASS_NAMES_DICT[class_id], "out": 0, "in": 0}
            for class_id in self.VEHICLE_CLASS_IDS
        }
        self.total_lp_detections = 0
        self.total_ocr_successes = 0
        self.total_ocr_attempts = 0
        
        # FPS tracking
        self.frame_times = []

    def _initialize_models(self, vehicle_model_path, lp_model_path):
        print("Loading vehicle detection model...")
        self.model = YOLO(vehicle_model_path)
        self.model.fuse()
        
        print("Loading license plate detection model...")
        self.lp_model = YOLO(lp_model_path)
        self.lp_model.fuse()
        
        self.CLASS_NAMES_DICT = self.model.model.names

        print("Initializing PaddleOCR...")
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            use_gpu=True,
            lang='en',
            det_db_thresh=0.3,
            det_db_box_thresh=0.5,
            rec_batch_num=6,
            show_log=False
        )
        print("✓ Initialization complete.")
    
    def _initialize_video_io(self):
        self.video_info = sv.VideoInfo.from_video_path(self.input_source)
        self.generator = sv.get_video_frames_generator(self.input_source)
        print(f"Video: {self.video_info.width}x{self.video_info.height}, "
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

    def _initialize_lpr_zone(self):
        print("Initializing LPR Zone...")
        frame_height = self.video_info.height
        frame_width = self.video_info.width
        mid_height = frame_height // 2 
        
        lpr_roi_polygon = np.array([
            [0, mid_height],
            [frame_width, mid_height],
            [frame_width, frame_height],
            [0, frame_height]
        ])
        
        self.lpr_zone = sv.PolygonZone(
            polygon=lpr_roi_polygon, 
            frame_resolution_wh=self.video_info.resolution_wh,
            triggering_anchors=[sv.Position.BOTTOM_CENTER]
        )

    def _initialize_annotators(self):
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
            display_in_count=True,
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

    #  ASYNC OCR WORKER
    def _ocr_worker(self):
        """Background thread xử lý OCR bất đồng bộ"""
        while self.ocr_thread_active:
            try:
                task = self.ocr_queue.get(timeout=0.1)
                if task is None:
                    break
                
                tracker_id, lp_crop, frame_index, class_id = task
                
                try:
                    # Thực hiện OCR và xử lý kết quả 
                    ocr_result = self.ocr.ocr(lp_crop, cls=False)
                    plate_text, ocr_conf = self._clean_ocr_text(ocr_result, class_id)
                    
                    self.ocr_result_queue.put((tracker_id, plate_text, ocr_conf, frame_index))
                except Exception as e:
                    print(f"[OCR Worker] Error for vehicle #{tracker_id}: {e}")
                    
            except queue.Empty:
                continue

    def _detect_vehicles(self, frame: np.ndarray) -> sv.Detections:
        results = self.model(frame, verbose=False, device=0)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[detections.confidence > self.VEHICLE_CONF_THRESHOLD]
        detections = detections[np.isin(detections.class_id, self.VEHICLE_CLASS_IDS)]
        return detections

    def _update_counts(self, tracked_detections: sv.Detections):
        if tracked_detections.tracker_id is None:
            return
        # Thực hiện phép so sánh hình học giữa Quá khứ và Hiện tại của từng chiếc xe.
        crossed_in, crossed_out = self.line_zone.trigger(tracked_detections)

        for i, (tracker_id, class_id) in enumerate(
            zip(tracked_detections.tracker_id, tracked_detections.class_id)
        ):
            if class_id in self.vehicle_counts:
                if crossed_out[i]:
                    self.vehicle_counts[class_id]["out"] += 1
                if crossed_in[i]:
                    self.vehicle_counts[class_id]["in"] += 1

    # =====================DÀNH CHO BIỂN SỐ VIỆT NAM=========================== 
    def _clean_ocr_text(self, ocr_result, class_id=None) -> Tuple[str, float]:
        if ocr_result is None or not ocr_result:
            return "", 0.0
        
        try:
            # 1. Trích xuất text thô & Confidence
            text_parts = []
            conf_parts = []
            if isinstance(ocr_result, list) and len(ocr_result) > 0:
                if isinstance(ocr_result[0], list) and len(ocr_result[0]) > 0:
                    for item in ocr_result[0]:
                        if len(item) >= 2:
                            text, confidence = item[1][0], item[1][1]
                            if text:
                                text_parts.append(str(text))
                                conf_parts.append(confidence)
            
            if not text_parts:
                return "", 0.0
            
            full_text = ''.join(text_parts)
            cleaned = re.sub(r'[^A-Z0-9]', '', full_text.upper())
            
            if len(cleaned) < 6: 
                return "", 0.0

            
            # Định nghĩa từ điển sửa lỗi
            dict_char_to_int = {'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8', 'D': '0', 'G': '6'}
            dict_int_to_char = {'0': 'D', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '4': 'A', '6': 'G'}
            
            text_list = list(cleaned)
            
            
            for i in [0, 1]:
                if i < len(text_list) and text_list[i] in dict_char_to_int:
                    text_list[i] = dict_char_to_int[text_list[i]]
            
            for i in range(len(text_list) - 4, len(text_list)):
                if i >= 0 and text_list[i] in dict_char_to_int:
                    text_list[i] = dict_char_to_int[text_list[i]]

            CAR_IDS = [2, 5, 7]
            MOTOR_IDS = [3]     
            
            if class_id in CAR_IDS:
                if len(text_list) > 2 and text_list[2] in dict_int_to_char:
                    text_list[2] = dict_int_to_char[text_list[2]]
                
                if 3 < len(text_list) and text_list[3] in dict_char_to_int:
                    text_list[3] = dict_char_to_int[text_list[3]]

            elif class_id in MOTOR_IDS:
                if len(text_list) > 2 and text_list[2] in dict_int_to_char:
                    text_list[2] = dict_int_to_char[text_list[2]]
                
                pass 

            final_text = "".join(text_list)
            avg_conf = sum(conf_parts) / len(conf_parts) if conf_parts else 0.0
            
            if avg_conf < self.OCR_CONFIDENCE_THRESHOLD:
                return "", 0.0

            return final_text, avg_conf
        
        except Exception as e:
            print(f"[Text Clean Error] {e}")
            return "", 0.0

    def _should_process_lpr(self, tracker_id: int, frame_index: int) -> bool:
        if tracker_id in self.confirmed_plates:
            return False
            
        if self.lpr_attempts.get(tracker_id, 0) >= self.MAX_LPR_ATTEMPTS:
            return False
            
        last_frame = self.last_lpr_frame.get(tracker_id, -999)
        if frame_index - last_frame < self.LPR_FRAME_INTERVAL:
            return False
            
        return True

    def _select_best_plate(self, tracker_id: int) -> Optional[str]:
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
        
        if best_text and tracker_id not in self.confirmed_plates:
            vote_count = Counter([txt for txt, conf in candidates])
            count = vote_count.get(best_text, 0)
            
            if count >= self.CONFIRMATION_THRESHOLD and best_score >= self.CONFIRMATION_SCORE_THRESHOLD:
                self.confirmed_plates.add(tracker_id)
                print(f" CONFIRMED: Vehicle #{tracker_id} = '{best_text}' "
                      f"(Score: {best_score:.2f}, Votes: {count})")
        
        return best_text

    def _process_lpr(self, frame: np.ndarray, tracked_detections: sv.Detections, frame_index: int):
        """ OPTIMIZED: Batch LP detection + Async OCR"""
        if tracked_detections.tracker_id is None:
            return
        
        try:
            mask = self.lpr_zone.trigger(detections=tracked_detections)
            detections_in_zone = tracked_detections[mask]
        except Exception:
            detections_in_zone = sv.Detections.empty()

        #  BATCH CROP: Thu thập tất cả crops cần xử lý
        crops_to_process = []
        crop_metadata = []
        
        for xyxy, conf, class_id, tracker_id in zip(
            detections_in_zone.xyxy,
            detections_in_zone.confidence,
            detections_in_zone.class_id,
            detections_in_zone.tracker_id
        ):
            # Kiểm tra xem biển số này đã confirmed hay chưa 
            if not self._should_process_lpr(tracker_id, frame_index):
                continue
            
            self.last_lpr_frame[tracker_id] = frame_index
            self.lpr_attempts[tracker_id] = self.lpr_attempts.get(tracker_id, 0) + 1
            # Cắt vùng xe (crop)
            x1, y1, x2, y2 = map(int, xyxy)
            pad = 10
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
            
            vehicle_crop = frame[y1:y2, x1:x2]
            if vehicle_crop.size == 0 or vehicle_crop.shape[0] < 20:
                continue
            
            crops_to_process.append(vehicle_crop)
            crop_metadata.append((tracker_id, x1, y1, x2, y2))
        
        if not crops_to_process:
            return
        
        # BATCH LP DETECTION
        try:
            lp_results_batch = self.lp_model(crops_to_process, verbose=False)
        except Exception as e:
            print(f"[Batch LP Detection] Error: {e}")
            return
        
        # Process từng kết quả
        for idx, (lp_results, (tracker_id, x1, y1, x2, y2)) in enumerate(
            zip(lp_results_batch, crop_metadata)
        ):
            try:
                lp_detections = sv.Detections.from_ultralytics(lp_results)
                lp_detections = lp_detections[lp_detections.confidence > self.LP_CONF_THRESHOLD]
                
                if len(lp_detections) == 0:
                    continue
                
                # Annotate raw detections
                vehicle_crop = crops_to_process[idx]
                abs_xyxy = lp_detections.xyxy.copy()
                abs_xyxy[:, 0] += x1; abs_xyxy[:, 1] += y1
                abs_xyxy[:, 2] += x1; abs_xyxy[:, 3] += y1
                abs_lp_det = sv.Detections(
                    xyxy=abs_xyxy, 
                    confidence=lp_detections.confidence, 
                    class_id=lp_detections.class_id
                )
                self.current_frame_lp_detections = sv.Detections.merge([
                    self.current_frame_lp_detections, abs_lp_det
                ])
                
                # Get best LP
                best_idx = np.argmax(lp_detections.confidence)
                relative_lp_box = lp_detections.xyxy[best_idx]
                self.total_lp_detections += 1
                # Cắt ảnh biển số
                lx1, ly1, lx2, ly2 = map(int, relative_lp_box)
                lp_crop = vehicle_crop[ly1:ly2, lx1:lx2]
                
                if lp_crop.size == 0 or lp_crop.shape[0] < 10:
                    continue
                
                #  ASYNC OCR: Đưa vào queue thay vì chờ
                self.total_ocr_attempts += 1
                try:
                    self.ocr_queue.put_nowait((tracker_id, lp_crop, frame_index, class_id))
                except queue.Full:
                    pass  # Skip nếu queue đầy
                
                # Lưu LP box để vẽ sau
                abs_lp_box = [
                    relative_lp_box[0] + x1, 
                    relative_lp_box[1] + y1,
                    relative_lp_box[2] + x1, 
                    relative_lp_box[3] + y1
                ]
                self.lp_boxes[tracker_id] = np.array(abs_lp_box)
                
            except Exception as e:
                print(f"[Vehicle #{tracker_id}] LP processing error: {e}")

    def _process_ocr_results(self):
        """Xử lý kết quả OCR từ queue"""
        while not self.ocr_result_queue.empty():
            try:
                tracker_id, plate_text, ocr_conf, frame_index = self.ocr_result_queue.get_nowait()
                
                if plate_text and len(plate_text) >= self.MIN_PLATE_LENGTH and ocr_conf > 0:
                    self.total_ocr_successes += 1
                    
                    if tracker_id not in self.plate_candidates:
                        self.plate_candidates[tracker_id] = []
                    
                    self.plate_candidates[tracker_id].append((plate_text, ocr_conf))
                    
                    best_plate = self._select_best_plate(tracker_id)
                    if best_plate:
                        self.plate_texts[tracker_id] = best_plate
                        
            except queue.Empty:
                break

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
        frame = self.line_zone_annotator.annotate(frame, line_counter=self.line_zone)
        frame = self.lpr_zone_annotator.annotate(frame, label="LPR Zone")

        # Vẽ confirmed plates
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


    def process(self):
        """
        Hàm chính xử lý video:
        - Đọc từng frame từ video input
        - Detect xe → Track → Count → LPR → Annotate
        - Ghi kết quả ra video output
        """
        total_start = time.time()
        frames_processed = 0
        
        try:
            with sv.VideoSink(target_path=self.output_source, video_info=self.video_info) as sink:
                for index, frame in enumerate(self.generator):
                    frame_start = time.time()

                    # Phát hiện phương tiện -> ByteTrack -> Cập nhật đếm
                    detections = self._detect_vehicles(frame)
                    tracked_detections = self.byte_tracker.update_with_detections(detections)
                    self._update_counts(tracked_detections)
                    
                    # Xử lý LPR mỗi LPR_FRAME_INTERVAL khung, Nếu không thì lấy kết quả OCR đã có
                    if index % self.LPR_FRAME_INTERVAL == 0:
                        self.current_frame_lp_detections = sv.Detections.empty()
                        self._process_lpr(frame, tracked_detections, index)
                    else:
                        self._process_ocr_results()
                    
                    # Annotate
                    annotated_frame = self._annotate_frame(frame.copy(), tracked_detections, index)
                    
                    # Tính FPS 
                    frame_time = time.time() - frame_start
                    self.frame_times.append(frame_time)
                    if len(self.frame_times) > 100:
                        self.frame_times.pop(0)
                    
                    proc_fps = len(self.frame_times) / sum(self.frame_times) if self.frame_times else 0
                    elapsed = time.time() - total_start
                    overall_fps = (index + 1) / elapsed if elapsed > 0 else 0
                    
                    cv2.putText(annotated_frame, f"FPS: {proc_fps:.1f}", 
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    sink.write_frame(frame=annotated_frame)
                    frames_processed += 1
                    
                    if index % 30 == 0:
                        print(f"Frame {index}/{self.video_info.total_frames} "
                            f"({100*index/self.video_info.total_frames:.1f}%) | FPS: {proc_fps:.2f}")
                    
                    if self.ENABLE_DISPLAY:
                        cv2.imshow("Processing", cv2.resize(annotated_frame, (1280, 720)))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                            
        finally:
            total_time = time.time() - total_start
            avg_fps = frames_processed / total_time
            realtime_ratio = (avg_fps / self.video_info.fps) * 100
            
            print(f"\n{'='*60}")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Frames Processed: {frames_processed}/{self.video_info.total_frames}")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Real-time Ratio: {realtime_ratio:.1f}%")
            print(f"Status: {' REAL-TIME' if realtime_ratio >= 100 else ' OFFLINE'}")
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
                    status = "[CONFIRMED]" if tracker_id in self.confirmed_plates else "[UNVERIFIED]"
                    log_message += f"Vehicle #{tracker_id}: {plate} {status}\n"
                log_message += f"\nTotal plates detected: {len(self.plate_texts)}\n"
            else:
                log_message += "No plates detected.\n"

            with open(self.log_file_path, 'w', encoding='utf-8') as f:
                f.write(log_message)
            print(f"\n Log saved successfully to {self.log_file_path}")

        except Exception as e:
            print(f"Error saving log file: {e}")

    def process_image(self):
        """Xử lý trên 1 ảnh tĩnh - Detect vehicles, license plates và OCR."""
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

        # 2. Loop qua từng xe để tìm biển số
        for i, (xyxy, conf, class_id) in enumerate(zip(
            vehicle_detections.xyxy, 
            vehicle_detections.confidence, 
            vehicle_detections.class_id
        )):
            vehicle_name = self.CLASS_NAMES_DICT.get(class_id, "Vehicle")
            plate_text = ""
            
            # Crop xe
            x1, y1, x2, y2 = map(int, xyxy)
            vehicle_crop = frame[y1:y2, x1:x2]
            
            if vehicle_crop.size > 0:
                # Detect Plate (YOLO LP)
                lp_results = self.lp_model(vehicle_crop, verbose=False, device=0)[0]
                lp_detections = sv.Detections.from_ultralytics(lp_results)
                lp_detections = lp_detections[lp_detections.confidence > self.LP_CONF_THRESHOLD]
                
                # Nếu tìm thấy biển số
                if len(lp_detections) > 0:
                    best_idx = np.argmax(lp_detections.confidence)
                    lpx1, lpy1, lpx2, lpy2 = map(int, lp_detections.xyxy[best_idx])
                    
                    lp_crop = vehicle_crop[lpy1:lpy2, lpx1:lpx2]
                    
                    if lp_crop.size > 0:
                        # OCR
                        try:
                            ocr_result = self.ocr.ocr(lp_crop, cls=False)
                            text, conf_ocr = self._clean_ocr_text(ocr_result, class_id)
                            
                            if text and len(text) >= self.MIN_PLATE_LENGTH:
                                plate_text = text
                                # Lưu tọa độ tuyệt đối của biển số để vẽ
                                abs_lp_box = [lpx1 + x1, lpy1 + y1, lpx2 + x1, lpy2 + y1]
                                lp_xyxy_list.append(abs_lp_box)
                                plate_labels.append(text)
                                lp_class_list.append(0)
                                print(f"  -> Vehicle {i+1}: Found Plate '{text}' (conf: {conf_ocr:.3f})")
                        except Exception as e:
                            print(f"  -> Vehicle {i+1}: OCR Error - {e}")

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
            lp_detections = sv.Detections(
                xyxy=np.array(lp_xyxy_list), 
                class_id=np.array(lp_class_list)
            )
            frame = self.lp_box_annotator.annotate(scene=frame, detections=lp_detections)
            frame = self.lp_label_annotator.annotate(scene=frame, detections=lp_detections, labels=plate_labels)

        # Vẽ thống kê đơn giản
        frame = self._draw_image_stats(frame, len(vehicle_detections), len(lp_xyxy_list))

        # 4. Lưu ảnh
        cv2.imwrite(self.output_source, frame)
        print(f"\n✓ Saved result to: {self.output_source}")
        print(f"  - Total Vehicles: {len(vehicle_detections)}")
        print(f"  - Total Plates Detected: {len(lp_xyxy_list)}")

    def _draw_image_stats(self, frame, vehicle_count, plate_count):
        """Vẽ bảng thống kê cho ảnh tĩnh."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (400, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        cv2.putText(frame, "ANALYSIS RESULT", (40, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Vehicles: {vehicle_count}", (40, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Plates: {plate_count}", (40, 130), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return frame

# --- ENTRY POINT ---
if __name__ == "__main__":
    
    # Đường dẫn cho các model nhận diện
    VEHICLE_MODEL_PATH = "model/vehicle_detector.pt"
    LP_MODEL_PATH = "model/lpr_detector.pt"
    

    MODE = "VIDEO" 
    if MODE == "VIDEO":
        INPUT = "assets/video/license-counting-video.mp4"
        OUTPUT = "assets/video/result.mp4"
    else:
        INPUT = "assets/img/test_img.png"  
        OUTPUT = "assets/img/result.png"  

    try:
        tracker = ObjectTracking(
            vehicle_model_path=VEHICLE_MODEL_PATH,
            lp_model_path=LP_MODEL_PATH,
            input_source=INPUT,
            output_source=OUTPUT
        )
        
        if MODE == "VIDEO":
            tracker.process() 
        else:
            tracker.process_image() 
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()