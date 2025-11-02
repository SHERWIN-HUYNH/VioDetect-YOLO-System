import cv2
from ultralytics import YOLO
import csv

# Load your custom YOLO models
vehicle_model = YOLO('model/vehicle_detector.pt')      # replace with your vehicle detection model path
license_model = YOLO('model/license_plate_detector.pt')      # replace with your license detection model path

# Initialize video capture (replace with your video path or 0 for webcam)
cap = cv2.VideoCapture('license-counting-video.mp4')  # or use 0 for webcam

# Open CSV file to save detection data
with open('detection_results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['frame', 'object_type', 'confidence', 'x1', 'y1', 'x2', 'y2'])  # CSV headers

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit loop if video ends or can't read frame
        
        frame_id += 1

        # Detect vehicles
        vehicle_results = vehicle_model(frame)
        for result in vehicle_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # Draw bounding box and label on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'Vehicle {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                # Write to CSV
                writer.writerow([frame_id, 'vehicle', conf, x1, y1, x2, y2])

        # Detect license plates
        license_results = license_model(frame)
        for result in license_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f'Plate {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                # Write to CSV
                writer.writerow([frame_id, 'license_plate', conf, x1, y1, x2, y2])

        # Display the frame with detections
        cv2.imshow('Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
