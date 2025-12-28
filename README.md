# VioDetect-YOLO-System

# Activate virtual environment
source bytrack-yolo/Scripts/activate
# Run code
python new_object_tracking.py
python gemini_speed.py
python current.py
python test.py
python research/test_model.py
python app/main.py
python app/setup_zones.py

# Determine the points of virtual line and polygon based on the video 
python extract_frame.py
python calculate_point.py