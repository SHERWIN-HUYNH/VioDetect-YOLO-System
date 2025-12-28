import os
from config import AppConfig
from core import TrafficMonitor

def main():
    # 1. Configuration
    config = AppConfig()
    config.ENABLE_DISPLAY = True 

    # 2. Setup Mode & Paths
    MODE = "VIDEO"  # "VIDEO" or "IMAGE"

    if MODE == "VIDEO":
        INPUT_FILE = "assets/video/vn_traffic_shorter.mp4"
        OUTPUT_FILE = "assets/video/result.mp4"
    else:
        INPUT_FILE = "assets/img/test_img.png"
        OUTPUT_FILE = "assets/img/result.png"

    # 3. Validation
    if not os.path.exists(INPUT_FILE):
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 4. Execution
    try:
        print(f"Initializing Traffic Monitor in {MODE} mode...")
        app = TrafficMonitor(config)
        
        if MODE == "VIDEO":
            app.process_video(INPUT_FILE, OUTPUT_FILE)
        elif MODE == "IMAGE":
            app.process_image(INPUT_FILE, OUTPUT_FILE)
            
        print("\nAll tasks completed successfully.")
        
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()