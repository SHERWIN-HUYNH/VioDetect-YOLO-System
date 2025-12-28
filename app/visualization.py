import cv2
import numpy as np
import supervision as sv
from typing import Dict, List, Optional
from config import AppConfig

class TrafficVisualizer:

    def __init__(self, config: AppConfig):
        self.cfg = config
        
        # --- Initialize Supervision Annotators ---
        # 1. Vehicle Annotators
        self.box_annotator = sv.BoundingBoxAnnotator(
            thickness=self.cfg.BOX_THICKNESS
        )
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=self.cfg.TEXT_THICKNESS, 
            text_scale=self.cfg.TEXT_SCALE
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=self.cfg.TRACE_THICKNESS, 
            trace_length=self.cfg.TRACE_LENGTH
        )
        
        # 2. Line Zone Annotator (Counting)
        self.line_zone_annotator = sv.LineZoneAnnotator(
            thickness=self.cfg.BOX_THICKNESS, 
            text_thickness=self.cfg.TEXT_THICKNESS,
            text_scale=self.cfg.TEXT_SCALE, 
            color=sv.Color.RED,
            display_in_count=True, 
            display_out_count=False
        )
        
        # 3. License Plate Annotators
        # Green for confirmed plates
        self.lp_box_annotator = sv.BoundingBoxAnnotator(
            thickness=self.cfg.BOX_THICKNESS, 
            color=sv.Color.BLUE
        )
        self.lp_label_annotator = sv.LabelAnnotator(
            text_thickness=self.cfg.TEXT_THICKNESS, 
            text_scale=self.cfg.TEXT_SCALE, 
            text_color=sv.Color.WHITE
        )
        # Yellow for raw detections (Debug)
        self.raw_lp_box_annotator = sv.BoundingBoxAnnotator(
            thickness=2, 
            color=sv.Color.YELLOW
        )
        
        # Will be initialized later when video resolution is known
        self.lpr_zone_annotator = None 

    def set_lpr_zone(self, zone: sv.PolygonZone):
        """
        Helper để setup LPR Zone Annotator sau khi có thông tin video.
        """
        self.lpr_zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone, 
            color=sv.Color.GREEN, 
            thickness=2,
            text_scale=1, 
            text_thickness=2, 
            text_padding=5
        )

    def draw_dashboard(self, frame: np.ndarray, frame_index: int, vehicle_counts: Dict) -> np.ndarray:
        """
        Hàm vẽ bảng thống kê. Logic cũ được refactor gọn gàng hơn.
        """
        overlay = frame.copy()
        h, w = frame.shape[:2]
        
        # Panel Settings
        PANEL_W, PANEL_H = 600, 500
        PANEL_X, PANEL_Y = w - 620, 20
        
        # Draw semi-transparent background
        cv2.rectangle(overlay, (PANEL_X, PANEL_Y), (PANEL_X + PANEL_W, PANEL_Y + PANEL_H), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Helper font params
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        WHITE = (255, 255, 255)
        ORANGE = (0, 100, 255)
        
        # Header
        cv2.putText(frame, "VEHICLE STATISTICS", (PANEL_X + 20, PANEL_Y + 60), 
                    FONT, 1.3, WHITE, 3)
        cv2.line(frame, (PANEL_X + 20, PANEL_Y + 80), (PANEL_X + PANEL_W - 20, PANEL_Y + 80), WHITE, 2)
        cv2.putText(frame, f"Frame: {frame_index}", (PANEL_X + 20, PANEL_Y + 120), 
                    FONT, 0.8, (200, 200, 200), 2)

        # Total Out
        total_out = sum(v["out"] for v in vehicle_counts.values())
        cv2.putText(frame, f"TOTAL OUT: {total_out}", (PANEL_X + 20, PANEL_Y + 180), 
                    FONT, 1.1, ORANGE, 3)

        # Details per class
        y_offset = PANEL_Y + 260
        # Color mapping logic from original code
        colors = {2: (255, 200, 0), 3: (255, 100, 255), 5: (0, 255, 255), 7: (100, 255, 100)}
        
        for cid, data in vehicle_counts.items():
            name = data["name"].upper()
            count = data["out"]
            color = colors.get(cid, WHITE)
            
            cv2.putText(frame, name, (PANEL_X + 20, y_offset), FONT, 1, color, 3)
            cv2.putText(frame, f"OUT: {count}", (PANEL_X + 200, y_offset), FONT, 1, ORANGE, 3)
            y_offset += 50
            
        return frame

    def draw_image_stats(self, frame: np.ndarray, vehicle_count: int, plate_count: int) -> np.ndarray:
        """
        Hàm vẽ thống kê cho chế độ ảnh tĩnh.
        """
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