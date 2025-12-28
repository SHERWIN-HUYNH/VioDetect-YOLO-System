import os
import cv2
import time
import datetime
import numpy as np
from collections import Counter
import supervision as sv
from ultralytics import YOLO

from config import AppConfig
from ocr import AsyncOCRManager
from visualization import TrafficVisualizer
from utils import PlateUtils

class TrafficMonitor:
    def __init__(self, config: AppConfig):
        self.cfg = config
        
        # Load Models
        print(f"Loading Models: {config.VEHICLE_MODEL_PATH} & {config.LP_MODEL_PATH}")
        self.vehicle_model = YOLO(config.VEHICLE_MODEL_PATH)
        self.vehicle_model.fuse()
        self.lp_model = YOLO(config.LP_MODEL_PATH)
        self.lp_model.fuse()
        self.vehicle_class_names = self.vehicle_model.model.names
        
        # Components
        self.ocr_manager = AsyncOCRManager(config)
        self.visualizer = TrafficVisualizer(config)
        
        # Tracking & Counting
        self.tracker = sv.ByteTrack(
            track_activation_threshold=config.TRACK_ACTIVATION_THRESHOLD,
            lost_track_buffer=config.LOST_TRACK_BUFFER,
            minimum_matching_threshold=config.MIN_MATCHING_THRESHOLD,
            frame_rate=config.FRAME_RATE
        )
        self.line_zone = sv.LineZone(
            start=config.LINE_START, 
            end=config.LINE_END,
            triggering_anchors=[sv.Position.BOTTOM_CENTER]
        )
        
        # State Management
        self.vehicle_counts = {
            cid: {"name": self.vehicle_class_names[cid], "out": 0, "in": 0}
            for cid in config.VEHICLE_CLASS_IDS
        }
        self.plate_data = {
            "texts": {},        
            "candidates": {},   
            "attempts": {},     
            "last_frame": {},   
            "confirmed": set(), 
            "boxes": {}         
        }
        self.current_raw_lp = sv.Detections.empty()
        self.crossed_vehicles = set()

        self.scale_x = 1.0
        self.scale_y = 1.0

    def _setup_zones_and_scale(self, original_wh, target_wh):
        """
        Tự động tính tỷ lệ và scale tọa độ từ Config (4K) sang Target.
        """
        orig_w, orig_h = original_wh
        target_w, target_h = target_wh
        
        # Tính tỷ lệ scale
        self.scale_x = target_w / orig_w
        self.scale_y = target_h / orig_h
        
        print(f"[AUTO-SCALE] {orig_w}x{orig_h} -> {target_w}x{target_h} (Factor: {self.scale_x:.3f})")

        # 2. Tự động Scale LINE ZONE
        s_start = sv.Point(
            int(self.cfg.LINE_START.x * self.scale_x), 
            int(self.cfg.LINE_START.y * self.scale_y)
        )
        s_end = sv.Point(
            int(self.cfg.LINE_END.x * self.scale_x), 
            int(self.cfg.LINE_END.y * self.scale_y)
        )
        self.line_zone = sv.LineZone(start=s_start, end=s_end, triggering_anchors=[sv.Position.BOTTOM_CENTER])

        # 3. Tự động Scale POLYGON ZONE
        if hasattr(self.cfg, 'LPR_ZONE_POLYGON') and self.cfg.LPR_ZONE_POLYGON:
            polygon_4k = np.array(self.cfg.LPR_ZONE_POLYGON)
        else:
            print("[WARNING] Không tìm thấy LPR_ZONE_POLYGON trong config, dùng mặc định.")
            polygon_4k = np.array([[0, 0], [orig_w, 0], [orig_w, orig_h], [0, orig_h]])

        polygon_target = (polygon_4k * [self.scale_x, self.scale_y]).astype(int)
        
        print(f"[AUTO-SCALE] Polygon Points ({len(polygon_target)} points) mapped to Target.")
        
        self.lpr_zone = sv.PolygonZone(
            polygon=polygon_target, 
            frame_resolution_wh=target_wh,
            triggering_anchors=[sv.Position.BOTTOM_CENTER]
        )
        
        if hasattr(self.visualizer, 'set_lpr_zone'):
            self.visualizer.set_lpr_zone(self.lpr_zone)

    def _detect_and_crop_lpr(self, frame_4k: np.ndarray, detections_target: sv.Detections, frame_index: int):
        if detections_target.tracker_id is None: return
        
        # Trigger zone trên tọa độ 720p
        mask = self.lpr_zone.trigger(detections=detections_target)
        zone_detections = detections_target[mask]
        # zone_detections = detections_target
        crops_4k = []
        metadata = []
        
        for xyxy_small, class_id, tracker_id in zip(zone_detections.xyxy, zone_detections.class_id, zone_detections.tracker_id):
            if tracker_id in self.plate_data["confirmed"]: continue
            if self.plate_data["attempts"].get(tracker_id, 0) >= self.cfg.MAX_LPR_ATTEMPTS: continue
            
            last_f = self.plate_data["last_frame"].get(tracker_id, -999)
            if frame_index - last_f < self.cfg.LPR_FRAME_INTERVAL: continue
            
            # --- MAP NGƯỢC: Target -> 4K ---
            x1 = int(xyxy_small[0] / self.scale_x)
            y1 = int(xyxy_small[1] / self.scale_y)
            x2 = int(xyxy_small[2] / self.scale_x)
            y2 = int(xyxy_small[3] / self.scale_y)
            
            pad = 10 
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(frame_4k.shape[1], x2 + pad), min(frame_4k.shape[0], y2 + pad)
            
            vehicle_crop = frame_4k[y1:y2, x1:x2]
            if vehicle_crop.size < 100: continue
            
            crops_4k.append(vehicle_crop)
            metadata.append((tracker_id, x1, y1, class_id))
            
            self.plate_data["attempts"][tracker_id] = self.plate_data["attempts"].get(tracker_id, 0) + 1
            self.plate_data["last_frame"][tracker_id] = frame_index

        if not crops_4k: return

        # Batch Detect LP
        try:
            lp_results = self.lp_model(crops_4k, verbose=False)
        except Exception: return

        for i, res in enumerate(lp_results):
            lp_dets = sv.Detections.from_ultralytics(res)
            lp_dets = lp_dets[lp_dets.confidence > self.cfg.LP_CONF_THRESHOLD]
            if len(lp_dets) == 0: continue
            
            tracker_id, vx1, vy1, v_class = metadata[i]
            
            best_idx = np.argmax(lp_dets.confidence)
            lxyxy = lp_dets.xyxy[best_idx]
            lx1, ly1, lx2, ly2 = map(int, lxyxy)
            
             #  LƯU BOX NGAY KHI DETECT, KHÔNG ĐỢI OCR 
            box_4k = np.array([vx1+lx1, vy1+ly1, vx1+lx2, vy1+ly2])
            box_target = box_4k * [self.scale_x, self.scale_y, self.scale_x, self.scale_y]
            self.plate_data["boxes"][tracker_id] = box_target.astype(int)

            lp_crop = crops_4k[i][ly1:ly2, lx1:lx2] # ảnh crop LPR
            # Gửi crop LPR vào OCR Queue
            if lp_crop.size > 50:
                self.ocr_manager.add_task(tracker_id, lp_crop, frame_index, v_class)
                
                # Tính box hiển thị trên Target
                # box_4k = np.array([vx1+lx1, vy1+ly1, vx1+lx2, vy1+ly2])
                # box_target = box_4k * [self.scale_x, self.scale_y, self.scale_x, self.scale_y]
                # self.plate_data["boxes"][tracker_id] = box_target.astype(int)

    def _update_vehicle_counts(self, detections: sv.Detections):
        if detections.tracker_id is None: return
        crossed_in, crossed_out = self.line_zone.trigger(detections)
        for i, (class_id, out_trig, in_trig) in enumerate(zip(detections.class_id, crossed_out, crossed_in)):
            if class_id in self.vehicle_counts:
                tracker_id = detections.tracker_id[i]
                vehicle_name = self.vehicle_class_names.get(class_id, "Unknown")
                plate = self.plate_data["texts"].get(tracker_id, "NO_PLATE")
                
                if tracker_id in self.crossed_vehicles:
                    continue
                if out_trig or in_trig:
                    self.crossed_vehicles.add(tracker_id)

                if out_trig:
                    self.vehicle_counts[class_id]["out"] += 1
                    # print(f"🔴 OUT | Vehicle #{tracker_id} | {vehicle_name} | Plate: {plate} ")
                
                if in_trig:
                    self.vehicle_counts[class_id]["in"] += 1
                    # print(f"🔵 IN  | Vehicle #{tracker_id} | {vehicle_name} | Plate: {plate} ")

    def _process_ocr_queue_results(self):
        """Xử lý kết quả OCR từ queue"""
        results = self.ocr_manager.get_results()
        
        for tracker_id, text, conf, _ in results:
            if not text: 
                continue
            
            # Thu thập candidates
            if tracker_id not in self.plate_data["candidates"]:
                self.plate_data["candidates"][tracker_id] = []
            self.plate_data["candidates"][tracker_id].append((text, conf))
            
            # Voting Logic
            best_text = None
            best_score = 0.0
            candidates = self.plate_data["candidates"][tracker_id]
            vote_map = Counter([t for t, c in candidates])
            
            for t, c in candidates:
                score = len(t) * c
                if score > best_score:
                    best_score = score
                    best_text = t
            
            # Confirmation
            if best_text and tracker_id not in self.plate_data["confirmed"]:
                votes = vote_map.get(best_text, 0)
                if (votes >= self.cfg.CONFIRMATION_THRESHOLD and 
                    best_score >= self.cfg.CONFIRMATION_SCORE_THRESHOLD):
                    self.plate_data["confirmed"].add(tracker_id)
                    print(f" >>> CONFIRMED #{tracker_id}: {best_text}")
            
            if best_text:
                self.plate_data["texts"][tracker_id] = best_text

    

    def _prepare_annotations(self, frame, detections, frame_index):
        labels = []
        if detections.tracker_id is not None:
            # active_ids = set(detections.tracker_id)
            # stale_ids = set(self.plate_data["boxes"].keys()) - active_ids
            # for tid in stale_ids:
            #     self.plate_data["boxes"].pop(tid, None)
            for _, class_id, tracker_id in zip(detections.confidence, detections.class_id, detections.tracker_id):
                name = self.vehicle_class_names.get(class_id, "UNK")
                plate = self.plate_data["texts"].get(tracker_id)
                status = "[OK]" if tracker_id in self.plate_data["confirmed"] else ""
                labels.append(f"#{tracker_id} | {plate} {status}" if plate else f"#{tracker_id} {name}")

        # Draw Tracks, Boxes, Labels
        frame = self.visualizer.trace_annotator.annotate(frame, detections)
        frame = self.visualizer.box_annotator.annotate(frame, detections)
        frame = self.visualizer.label_annotator.annotate(frame, detections, labels)
        
        # Draw Zones & Dashboard
        # frame = self.visualizer.draw_dashboard(frame, frame_index, self.vehicle_counts)
        frame = self.visualizer.line_zone_annotator.annotate(frame, self.line_zone)
        if self.visualizer.lpr_zone_annotator:
            frame = self.visualizer.lpr_zone_annotator.annotate(frame, label="LPR Zone")
        
        # Draw Raw & Confirmed LPs
        frame = self.visualizer.raw_lp_box_annotator.annotate(frame, self.current_raw_lp)
        
        lp_boxes = [box for tid, box in self.plate_data["boxes"].items() if tid in detections.tracker_id]
        # lp_labels = [self.plate_data["texts"].get(tid, "") for tid in self.plate_data["boxes"] if tid in detections.tracker_id]
        
        if lp_boxes:
            lp_dets = sv.Detections(xyxy=np.array(lp_boxes), class_id=np.array([0]*len(lp_boxes)))
            frame = self.visualizer.lp_box_annotator.annotate(frame, lp_dets)
        #     frame = self.visualizer.lp_label_annotator.annotate(frame, lp_dets, lp_labels)
            
        return frame
    
    def process_video(self, input_path: str, output_path: str):
        print(f"--- PROCESSING HYBRID (FAST): {input_path} ---")
        
        video_info_4k = sv.VideoInfo.from_video_path(input_path)
        TARGET_W, TARGET_H = self.cfg.TARGET_WIDTH, self.cfg.TARGET_HEIGHT
        
        self._setup_zones_and_scale((video_info_4k.width, video_info_4k.height), (TARGET_W, TARGET_H))
        
        output_info = sv.VideoInfo(width=TARGET_W, height=TARGET_H,fps=self.cfg.FRAME_RATE)
        
        frames_processed = 0
        start_time = time.time()
        
        try:
            with sv.VideoSink(output_path, output_info) as sink:
                for index, frame_4k in enumerate(sv.get_video_frames_generator(input_path)):
                    
                    # 1. Resize NHANH 
                    frame_processing = cv2.resize(frame_4k, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
                    
                    # 2. Detect & Track trên ảnh nhỏ
                    results = self.vehicle_model(frame_processing, verbose=False, device=0)[0]
                    dets = sv.Detections.from_ultralytics(results)
                    dets = dets[dets.confidence > self.cfg.VEHICLE_CONF_THRESHOLD]
                    dets = dets[np.isin(dets.class_id, self.cfg.VEHICLE_CLASS_IDS)]
                    
                    tracked_dets = self.tracker.update_with_detections(dets)
                    
                    # 3. Logic Đếm & OCR Queue
                    self._update_vehicle_counts(tracked_dets)
                    self._process_ocr_queue_results() 
                    
                    # 4. Hybrid LPR (Crop từ 4K)
                    if index % self.cfg.LPR_FRAME_INTERVAL == 0:
                        self._detect_and_crop_lpr(frame_4k, tracked_dets, index)
                    
                    # 5. Visualize & Write
                    if index % self.cfg.FRAME_SKIP == 0:
                        annotated_frame = self._prepare_annotations(frame_processing, tracked_dets, index)
                        sink.write_frame(annotated_frame)
                        
                        if self.cfg.ENABLE_DISPLAY:
                            cv2.imshow("Monitor", annotated_frame) 
                            if cv2.waitKey(1) & 0xFF == ord('q'): break
                    
                    frames_processed += 1
                    if index % 30 == 0:
                        elapsed = time.time() - start_time
                        fps = frames_processed / elapsed if elapsed > 0 else 0
                        print(f"Frame {index} | Speed: {fps:.2f} FPS")

        except Exception as e:
            print(f"Error: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.ocr_manager.stop()
            self._save_log(frames_processed, start_time)
            cv2.destroyAllWindows()

    def process_image(self, input_path: str, output_path: str):
        print(f"--- PROCESSING IMAGE: {input_path} ---")
        frame = cv2.imread(input_path)
        if frame is None: return

        # Detect Vehicles
        results = self.vehicle_model(frame, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(results)
        dets = dets[dets.confidence > self.cfg.VEHICLE_CONF_THRESHOLD]
        dets = dets[np.isin(dets.class_id, self.cfg.VEHICLE_CLASS_IDS)]
        
        final_labels, lp_boxes, lp_labels = [], [], []

        # Process Detections
        for i, (xyxy, cls) in enumerate(zip(dets.xyxy, dets.class_id)):
            v_name = self.vehicle_class_names.get(cls, "Vehicle")
            plate_text = ""
            x1, y1, x2, y2 = map(int, xyxy)
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                # Detect & OCR License Plate
                lp_res = self.lp_model(crop, verbose=False)[0]
                lp_dets = sv.Detections.from_ultralytics(lp_res)
                lp_dets = lp_dets[lp_dets.confidence > self.cfg.LP_CONF_THRESHOLD]
                
                if len(lp_dets) > 0:
                    best = np.argmax(lp_dets.confidence)
                    lx1, ly1, lx2, ly2 = map(int, lp_dets.xyxy[best])
                    lp_crop = crop[ly1:ly2, lx1:lx2]
                    
                    if lp_crop.size > 0:
                        ocr_res = self.ocr_manager.ocr_engine.ocr(lp_crop, cls=False)
                        text, _ = PlateUtils.clean_ocr_text(ocr_res, cls, self.cfg.OCR_CONFIDENCE_THRESHOLD)
                        if text:
                            plate_text = text
                            lp_boxes.append([lx1+x1, ly1+y1, lx2+x1, ly2+y1])
                            lp_labels.append(text)

            label = f"{v_name} | {plate_text}" if plate_text else v_name
            final_labels.append(label)

        # Draw & Save
        frame = self.visualizer.box_annotator.annotate(frame, dets)
        frame = self.visualizer.label_annotator.annotate(frame, dets, final_labels)
        
        if lp_boxes:
            lp_dets = sv.Detections(xyxy=np.array(lp_boxes), class_id=np.array([0]*len(lp_boxes)))
            frame = self.visualizer.lp_box_annotator.annotate(frame, lp_dets)
            frame = self.visualizer.lp_label_annotator.annotate(frame, lp_dets, lp_labels)
        
        frame = self.visualizer.draw_image_stats(frame, len(dets), len(lp_boxes))
        cv2.imwrite(output_path, frame)
        self.ocr_manager.stop()

    def _save_log(self, frames_processed, start_time):
        """Save final counts and detected plates to log file."""
        try:
            duration = time.time() - start_time
            avg_fps = frames_processed / duration if duration > 0 else 0
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            log_message = f"[{timestamp}] - Processing Complete\n"
            log_message += f"=" * 60 + "\n"
            
            # Video Processing Stats
            log_message += f"\n=== PROCESSING STATISTICS ===\n"
            log_message += f"Total Frames Processed: {frames_processed}\n"
            log_message += f"Total Duration: {duration:.2f}s\n"
            log_message += f"Average FPS: {avg_fps:.2f}\n"
            log_message += f"Real-time Ratio: {(avg_fps / self.cfg.FRAME_RATE * 100):.1f}%\n"
            
            # Vehicle Counts
            log_message += f"\n=== VEHICLE COUNTS ===\n"
            total_out = 0
            total_in = 0
            for class_id, data in self.vehicle_counts.items():
                name = data['name'].upper()
                out_count = data['out']
                in_count = data['in']
                log_message += f"{name}: OUT={out_count}, IN={in_count}\n"
                total_out += out_count
                total_in += in_count
            log_message += "-" * 40 + "\n"
            log_message += f"TOTAL: OUT={total_out}, IN={total_in}\n"
            
            # LPR Statistics (giống test.py)
            log_message += f"\n=== LPR STATISTICS ===\n"
            total_lp_detections = len([tid for tid, box in self.plate_data["boxes"].items()])
            total_ocr_attempts = sum(self.plate_data["attempts"].values())
            total_ocr_successes = len([tid for tid, text in self.plate_data["texts"].items() if text])
            total_confirmed = len(self.plate_data["confirmed"])
            
            log_message += f"Total YOLO-LP Detections (Events): {total_lp_detections}\n"
            log_message += f"Total OCR Attempts (Queue): {total_ocr_attempts}\n"
            log_message += f"Total Successful OCR Reads: {total_ocr_successes}\n"
            log_message += f"Total Plates Confirmed: {total_confirmed}\n"
            
            # Detected Plates - HIỂN THỊ TẤT CẢ (cả confirmed và unverified)
            log_message += f"\n=== DETECTED PLATES ===\n"
            if self.plate_data["texts"]:
                for tracker_id in sorted(self.plate_data["texts"].keys()):
                    plate_text = self.plate_data["texts"][tracker_id]
                    status = "[CONFIRMED]" if tracker_id in self.plate_data["confirmed"] else "[UNVERIFIED]"
                    attempts = self.plate_data["attempts"].get(tracker_id, 0)
                    candidates_count = len(self.plate_data["candidates"].get(tracker_id, []))
                    
                    log_message += f"Vehicle #{tracker_id}: {plate_text} {status} "
                    log_message += f"(Attempts: {attempts}, Candidates: {candidates_count})\n"
                
                log_message += f"\nTotal plates detected: {len(self.plate_data['texts'])}\n"
                log_message += f"├── Confirmed: {total_confirmed}\n"
                log_message += f"└── Unverified: {len(self.plate_data['texts']) - total_confirmed}\n"
            else:
                log_message += "No plates detected.\n"
            
            log_message += "\n" + "=" * 60 + "\n"
            
            # Write to file
            with open(self.cfg.LOG_FILE, 'w', encoding='utf-8') as f:
                f.write(log_message)
            
            print(f"\n✓ Log saved successfully to {self.cfg.LOG_FILE}")
            print(f"  Total Vehicles: OUT={total_out}, IN={total_in}")
            print(f"  Total Plates: {len(self.plate_data['texts'])} ({total_confirmed} confirmed)")
            
        except Exception as e:
            print(f"[ERROR] Failed to save log: {e}")
            import traceback
            traceback.print_exc()