from dataclasses import dataclass, field
from typing import List
import supervision as sv

@dataclass
class AppConfig:
    # --- MODEL PATHS ---
    VEHICLE_MODEL_PATH: str = "model/vehicle_detector.pt"
    LP_MODEL_PATH: str = "model/new_lpr_detector.pt"
    LOG_FILE: str = "log.txt"

    # --- DETECTION SETTINGS ---
    
    VEHICLE_CONF_THRESHOLD: float = 0.4
    LP_CONF_THRESHOLD: float = 0.3

    # --- TRACKING SETTINGS ---
    TRACK_ACTIVATION_THRESHOLD: float = 0.4
    LOST_TRACK_BUFFER: int = 120
    MIN_MATCHING_THRESHOLD: float = 0.8
    FRAME_RATE: int = 30

    # --- OCR & LOGIC SETTINGS ---
    OCR_CONFIDENCE_THRESHOLD: float = 0.15
    MIN_PLATE_LENGTH: int = 3
    MAX_LPR_ATTEMPTS: int = 150
    CONFIRMATION_THRESHOLD: int = 2
    CONFIRMATION_SCORE_THRESHOLD: float = 5.0
    
    # --- PERFORMANCE ---
    FRAME_SKIP: int = 1
    LPR_FRAME_INTERVAL: int = 3 # Thực hiện LPR mỗi n Frame 1 lần 
    OCR_QUEUE_SIZE: int = 30 # Đường đông thì giá trị này nên lớn hơn
    ENABLE_DISPLAY: bool = False  

    # --- ZONE CONFIGURATION (Counting Line) ---
    # Tọa độ trên ảnh 4k
    LINE_START: sv.Point = field(default_factory=lambda: sv.Point(1170, 891))
    LINE_END: sv.Point = field(default_factory=lambda: sv.Point(1800, 2151))
    LPR_ZONE_POLYGON: List[List[int]] = field(default_factory=lambda: [
            [1170, 891],    
            [3576, 624],   
            [3828, 1413],   
            [1800, 2151]    
        ])
    # LINE_START: sv.Point = field(default_factory=lambda: sv.Point(229, 708))
    # LINE_END: sv.Point = field(default_factory=lambda: sv.Point(1188, 1069))
    # LPR_ZONE_POLYGON: List[List[int]] = field(default_factory=lambda: [
    #     [229, 708],     # LINE_START
    #     [1285, 600],    # ZONE_P2
    #     [1896, 697],   # ZONE_P1
    #     [1188, 1069]    # LINE_END
    # ])

    # --- VISUALIZATION STYLES ---
    BOX_THICKNESS: int = 2
    TEXT_THICKNESS: int = 2
    TEXT_SCALE: float = 0.8
    TRACE_THICKNESS: int = 2
    TRACE_LENGTH: int = 10

    # --- VIDEO OUTPUT SETTINGS ---
    # OUTPUT_WIDTH: int = 1920   
    # OUTPUT_HEIGHT: int = 1080
    MAINTAIN_ASPECT_RATIO: bool = True  

    TARGET_WIDTH: int = 1280  
    TARGET_HEIGHT: int = 720
    VEHICLE_CLASS_IDS: List[int] = field(default_factory=lambda: [2, 3, 5, 7])