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
    Optimized class for vehicle tracking, counting, and license plate recognition.
    """
    
    # --- CONFIGURATION ---
    VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    VEHICLE_CONF_THRESHOLD = 0.3
    
    # LP detection threshold
    LP_CONF_THRESHOLD = 0.4
    
    # OCR configuration
    OCR_CONFIDENCE_THRESHOLD = 0.5
    
    # LineZone configuration - Single line for counting OUT
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
    FRAME_SKIP = 1  # Process every frame
    LPR_FRAME_INTERVAL = 5  # Run LPR every 5 frames per vehicle
    MAX_LPR_ATTEMPTS = 3  # Max attempts to read plate per vehicle

    
    def __init__(self, vehicle_model_path, lp_model_path, input_video_path, output_video_path, log_file_path):
        """Initialize the object tracking system."""
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.log_file_path = log_file_path

        self._initialize_models(vehicle_model_path, lp_model_path)
        self._initialize_video_io()
        self._initialize_tracker()
        self._initialize_zone()
        self._initialize_annotators()

        # State tracking
        self.plate_texts: Dict[int, str] = {}
        self.lpr_attempts: Dict[int, int] = {}
        self.last_lpr_frame: Dict[int, int] = {}
        self.plate_candidates: Dict[int, List[str]] = {}
        
        self.vehicle_counts = {
            class_id: {"name": self.CLASS_NAMES_DICT[class_id], "out": 0}
            for class_id in self.VEHICLE_CLASS_IDS
        }

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
            use_textline_orientation=True, 
            lang='en',
            text_det_thresh=0.3,
            text_det_box_thresh=0.5,
            text_recognition_batch_size=6
        )
        print("Initialization complete.")
    
    def _initialize_video_io(self):
        """Initialize video input/output with proper settings."""
        self.video_info = sv.VideoInfo.from_video_path(self.input_video_path)
        self.generator = sv.get_video_frames_generator(self.input_video_path)
        
        print(f"Video Info: {self.video_info.width}x{self.video_info.height}, "
              f"{self.video_info.fps} FPS, {self.video_info.total_frames} frames")

    def _initialize_tracker(self):
        """Initialize ByteTrack tracker."""
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=self.TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=self.LOST_TRACK_BUFFER,
            minimum_matching_threshold=self.MIN_MATCHING_THRESHOLD,
            frame_rate=self.FRAME_RATE
        )

    def _initialize_zone(self):
        """Initialize LineZone for counting."""
        self.line_zone = sv.LineZone(
            start=self.LINE_START,
            end=self.LINE_END,
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
            color=sv.Color.RED
        )

    def _detect_vehicles(self, frame: np.ndarray) -> sv.Detections:
        """Detect vehicles using YOLO model."""
        results = self.model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        # Lọc những phương tiện có độ tin cậy cao và thuộc các lớp mục tiêu
        detections = detections[detections.confidence > self.VEHICLE_CONF_THRESHOLD]
        detections = detections[np.isin(detections.class_id, self.VEHICLE_CLASS_IDS)]
        
        return detections

    def _update_counts(self, tracked_detections: sv.Detections):
        """Update vehicle counts when crossing the OUT line."""
        if tracked_detections.tracker_id is None:
            return
        
        _, crossed_out = self.line_zone.trigger(tracked_detections)
        
        for i, (tracker_id, class_id) in enumerate(
            zip(tracked_detections.tracker_id, tracked_detections.class_id)
        ):
            if class_id in self.vehicle_counts:
                if crossed_out[i]:
                    self.vehicle_counts[class_id]["out"] += 1
                    print(f"  ← Vehicle #{tracker_id} ({self.CLASS_NAMES_DICT[class_id]}) crossed OUT")

    def _preprocess_plate_image(self, plate_img: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for better OCR accuracy."""
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img
        
        h, w = gray.shape
        if h < 50 or w < 100:
            scale = max(50/h, 100/w)
            new_w, new_h = int(w * scale), int(h * scale)
            gray = cv2.resize(gray, (new_w, new_h))
        
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        denoised = cv2.fastNlMeansDenoising(binary)
        return denoised

    def _clean_ocr_text(self, ocr_result) -> str:
        """Extract and clean text from PaddleOCR results with robust error handling."""
        if ocr_result is None or not ocr_result:
            return ""
        
        try:
            text_parts = []
            
            if isinstance(ocr_result, list):
                if len(ocr_result) == 0:
                    return ""
                
                # Format: [[bbox, (text, confidence)], ...]
                if isinstance(ocr_result[0], list):
                    for item in ocr_result[0]:
                        if len(item) >= 2 and isinstance(item[1], (tuple, list)) and len(item[1]) >= 2:
                            text, confidence = item[1][0], item[1][1]
                            if text and confidence > self.OCR_CONFIDENCE_THRESHOLD:
                                text_parts.append(str(text))
                
                # Format: [(text, confidence), ...]
                elif isinstance(ocr_result[0], (tuple, list)) and len(ocr_result[0]) >= 2:
                    for item in ocr_result:
                        if len(item) >= 2:
                            text, confidence = item[0], item[1]
                            if text and confidence > self.OCR_CONFIDENCE_THRESHOLD:
                                text_parts.append(str(text))
                
                # Format: Direct text list
                elif isinstance(ocr_result[0], str):
                    text_parts = [str(item) for item in ocr_result]
            
            elif isinstance(ocr_result, str):
                text_parts = [ocr_result]
            
            if not text_parts:
                return ""
            
            text = ''.join(text_parts)
            text = re.sub(r'[^A-Z0-9]', '', text.upper())
            
            corrections = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8'}
            for old, new in corrections.items():
                text = text.replace(old, new)
            
            return text
            
        except Exception:
            return ""

    def _get_consensus_plate(self, tracker_id: int) -> Optional[str]:
        """Get consensus plate text from multiple readings."""
        if tracker_id not in self.plate_candidates:
            return None
        
        candidates = self.plate_candidates[tracker_id]
        if not candidates:
            return None
        
        counter = Counter(candidates)
        most_common = counter.most_common(1)[0]
        
        if most_common[1] >= 2 or len(most_common[0]) >= 6:
            return most_common[0]
        
        return None

    def _should_process_lpr(self, tracker_id: int, frame_index: int) -> bool:
        """Determine if LPR should be processed for this vehicle."""
        if tracker_id in self.plate_texts:
            return False
        
        if self.lpr_attempts.get(tracker_id, 0) >= self.MAX_LPR_ATTEMPTS:
            return False
        
        last_frame = self.last_lpr_frame.get(tracker_id, -999)
        if frame_index - last_frame < self.LPR_FRAME_INTERVAL:
            return False
        
        return True

    def _process_lpr(self, frame: np.ndarray, tracked_detections: sv.Detections, frame_index: int):
        """Optimized LPR pipeline with better plate detection."""
        if tracked_detections.tracker_id is None:
            return
        
        # Fixed: Proper zip without mask variable
        for xyxy, conf, class_id, tracker_id in zip(
            tracked_detections.xyxy,
            tracked_detections.confidence,
            tracked_detections.class_id,
            tracked_detections.tracker_id
        ):
            
            if not self._should_process_lpr(tracker_id, frame_index):
                continue
            
            try:
                self.last_lpr_frame[tracker_id] = frame_index
                self.lpr_attempts[tracker_id] = self.lpr_attempts.get(tracker_id, 0) + 1
                
                x1, y1, x2, y2 = map(int, xyxy)
                pad = 10
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(frame.shape[1], x2 + pad), min(frame.shape[0], y2 + pad)
                
                vehicle_crop = frame[y1:y2, x1:x2]
                if vehicle_crop.size == 0:
                    continue

                lp_results = self.lp_model(vehicle_crop, verbose=False)[0]
                lp_detections = sv.Detections.from_ultralytics(lp_results)
                lp_detections = lp_detections[lp_detections.confidence > self.LP_CONF_THRESHOLD]

                if len(lp_detections) > 0:
                    best_idx = np.argmax(lp_detections.confidence)
                    lx1, ly1, lx2, ly2 = map(int, lp_detections.xyxy[best_idx])
                    
                    lp_crop = vehicle_crop[ly1:ly2, lx1:lx2]
                    if lp_crop.size == 0:
                        continue
                    
                    processed_plate = self._preprocess_plate_image(lp_crop)
                    
                    try:
                        ocr_result1 = self.ocr.predict(lp_crop)
                    except Exception:
                        ocr_result1 = None
                    
                    try:
                        ocr_result2 = self.ocr.predict(processed_plate)
                    except Exception:
                        ocr_result2 = None
                    
                    plate_text1 = self._clean_ocr_text(ocr_result1) if ocr_result1 else ""
                    plate_text2 = self._clean_ocr_text(ocr_result2) if ocr_result2 else ""
                    
                    plate_text = plate_text1 if len(plate_text1) >= len(plate_text2) else plate_text2
                    
                    if plate_text and len(plate_text) >= 4:
                        if tracker_id not in self.plate_candidates:
                            self.plate_candidates[tracker_id] = []
                        self.plate_candidates[tracker_id].append(plate_text)
                        
                        consensus = self._get_consensus_plate(tracker_id)
                        if consensus:
                            self.plate_texts[tracker_id] = consensus
                            print(f"✓ Frame {frame_index}: Vehicle #{tracker_id} -> Plate: {consensus}")
            
            except Exception as e:
                print(f"Error processing LPR for tracker {tracker_id}: {e}")

    def _annotate_frame(self, frame: np.ndarray, tracked_detections: sv.Detections, 
                       frame_index: int) -> np.ndarray:
        """Annotate frame with detection results."""
        labels = []
        if tracked_detections.tracker_id is not None:
            for confidence, class_id, tracker_id in zip(
                tracked_detections.confidence, 
                tracked_detections.class_id, 
                tracked_detections.tracker_id
            ):
                vehicle_name = self.CLASS_NAMES_DICT.get(class_id, "Unknown")
                plate_text = self.plate_texts.get(tracker_id)
                
                if plate_text:
                    label = f"#{tracker_id} | {plate_text}"
                else:
                    label = f"#{tracker_id} {vehicle_name.upper()}"
                labels.append(label)

            frame = self.trace_annotator.annotate(scene=frame, detections=tracked_detections)
            frame = self.box_annotator.annotate(scene=frame, detections=tracked_detections)
            frame = self.label_annotator.annotate(
                scene=frame, detections=tracked_detections, labels=labels
            )
        
        frame = self.draw_statistics(frame, frame_index)
        frame = self.line_zone_annotator.annotate(frame, line_counter=self.line_zone)
        
        return frame

    def callback(self, frame: np.ndarray, index: int) -> np.ndarray:
        """Main callback for processing each frame."""
        detections = self._detect_vehicles(frame)
        tracked_detections = self.byte_tracker.update_with_detections(detections)
        self._update_counts(tracked_detections)
        self._process_lpr(frame, tracked_detections, index)
        annotated_frame = self._annotate_frame(frame.copy(), tracked_detections, index)
        return annotated_frame

    def process(self):
        """Main processing loop - optimized for speed and completeness."""
        frame_count = 0
        processed_count = 0
        
        try:
            with sv.VideoSink(target_path=self.output_video_path, video_info=self.video_info) as sink:
                for index, frame in enumerate(self.generator):
                    frame_count += 1
                    
                    if index % self.FRAME_SKIP != 0:
                        sink.write_frame(frame=frame)
                        continue
                    
                    if index % 30 == 0:
                        print(f"Processing frame {index}/{self.video_info.total_frames} "
                              f"({100*index/self.video_info.total_frames:.1f}%)")
                    
                    result_frame = self.callback(frame, index)
                    sink.write_frame(frame=result_frame)
                    processed_count += 1

                    if index % 3 == 0:
                        h, w = result_frame.shape[:2]
                        display_width = w // 3
                        display_height = h // 3
                        
                        if display_width > 0 and display_height > 0:
                            display_frame = cv2.resize(result_frame, (display_width, display_height))
                            cv2.imshow("Vehicle Tracking & LPR (Press 'q' to quit)", display_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nUser pressed 'q', stopping processing...")
                        break
                        
        except Exception as e:
            print(f"\nError during processing: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            print(f"\nProcessing complete: {processed_count}/{frame_count} frames processed")
            print("Saving log...")
            self.save_log()

    def draw_statistics(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Draw statistics panel on frame."""
        overlay = frame.copy()
        height, width = frame.shape[:2]

        PANEL_WIDTH, PANEL_HEIGHT = 600, 500
        PANEL_MARGIN = 20
        PANEL_X = width - PANEL_WIDTH - PANEL_MARGIN
        PANEL_Y = PANEL_MARGIN
        ALPHA = 0.7
        
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        SCALE_HEADER = 1.3
        SCALE_BODY = 1.1
        SCALE_SMALL = 0.8
        THICK_HEADER = 3
        THICK_BODY = 3
        THICK_SMALL = 2
        
        Y_HEADER = PANEL_Y + 60
        Y_LINE = PANEL_Y + 80
        Y_FRAME = PANEL_Y + 120
        Y_TOTAL = PANEL_Y + 180
        Y_DETAIL_START = PANEL_Y + 260
        Y_DETAIL_STEP = 50

        cv2.rectangle(
            overlay, 
            (PANEL_X, PANEL_Y), 
            (PANEL_X + PANEL_WIDTH, PANEL_Y + PANEL_HEIGHT), 
            (0, 0, 0), -1
        )
        frame = cv2.addWeighted(overlay, ALPHA, frame, 1 - ALPHA, 0)

        cv2.putText(frame, "VEHICLE STATISTICS", (PANEL_X + 20, Y_HEADER), 
                   FONT, SCALE_HEADER, (255, 255, 255), THICK_HEADER)
        cv2.line(frame, (PANEL_X + 20, Y_LINE), 
                (PANEL_X + PANEL_WIDTH - 20, Y_LINE), (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_index}", (PANEL_X + 20, Y_FRAME), 
                   FONT, SCALE_SMALL, (200, 200, 200), THICK_SMALL)

        total_out = sum(v["out"] for v in self.vehicle_counts.values())
        cv2.putText(frame, f"TOTAL OUT: {total_out}", (PANEL_X + 20, Y_TOTAL), 
                   FONT, SCALE_BODY, (0, 100, 255), THICK_BODY)

        y_offset = Y_DETAIL_START
        colors = {2: (255, 200, 0), 3: (255, 100, 255), 
                 5: (0, 255, 255), 7: (100, 255, 100)}
        
        for class_id in self.VEHICLE_CLASS_IDS:
            vehicle = self.vehicle_counts[class_id]
            color = colors.get(class_id, (255, 255, 255))
            name = vehicle["name"].upper()
            
            cv2.putText(frame, name, (PANEL_X + 20, y_offset), FONT, 1, color, THICK_BODY)
            cv2.putText(frame, f"OUT: {vehicle['out']}", (PANEL_X + 200, y_offset), 
                       FONT, 1, (0, 100, 255), THICK_BODY)
            y_offset += Y_DETAIL_STEP

        return frame

    def save_log(self):
        """Save final counts and detected plates to log file."""
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_message = f"[{timestamp}] - Processing Complete\n"
            log_message += f"Input Video: {self.input_video_path}\n"
            log_message += f"Output Video: {self.output_video_path}\n"
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
    LP_MODEL_PATH = "model/license_plate_detector.pt"
    
    INPUT_PATH = "assets/video/license-counting-video.mp4"
    OUTPUT_PATH = "assets/video/result-with-lpr-improved.mp4"
    LOG_PATH = "vehicle_count_lpr_log_improved.txt"

    try:
        print("="*60)
        print("VEHICLE TRACKING & LICENSE PLATE RECOGNITION")
        print("="*60)
        
        tracker = ObjectTracking(
            vehicle_model_path=VEHICLE_MODEL_PATH,
            lp_model_path=LP_MODEL_PATH,
            input_video_path=INPUT_PATH,
            output_video_path=OUTPUT_PATH,
            log_file_path=LOG_PATH
        )
        
        print("\nStarting video processing...")
        tracker.process()
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETE!")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: File not found. Please check the path: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()