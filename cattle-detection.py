import cv2
import torch
from ultralytics import YOLO
import time  # Import time module for delay
import datetime  # Import for timestamp

import firebase_admin
from firebase_admin import credentials, db

cred = credentials.Certificate("credentials.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://cattle-detection-project-default-rtdb.asia-southeast1.firebasedatabase.app"
})

ref = db.reference("/")

model = YOLO()

# Define object classes of interest
FARM_ANIMALS = ["cow", "goat", "horse"]
DANGEROUS_ANIMALS = ["lion", "tiger", "dog", "fox"]
HUMAN = "person"

# Initialize tracking variables
cow_count, goat_count, horse_count = 0, 0, 0
warning, danger = False, False
is_danger = False
danger_animal = None

# Open webcam
cap = cv2.VideoCapture(0)

last_update_time = time.time()  # Track the last update time

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Check if 3 seconds have passed since the last update
    if time.time() - last_update_time < 3:
        cv2.imshow("Farm Animal Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    last_update_time = time.time()  # Update the last update time

    # Run YOLOv8 inference
    results = model(frame, stream=True)
    
    # Reset counts and flags each frame
    cow_count, goat_count, horse_count = 0, 0, 0
    warning, danger = False, False
    danger_animal = None

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = box.conf[0].item()
            if conf < 0.3:  # Ignore low confidence detections
                continue

            label = model.names[cls_id]

            # Count farm animals
            if label == "cow":
                cow_count += 1
            elif label == "goat":
                goat_count += 1
            elif label == "horse":
                horse_count += 1

            # Check for humans
            if label == HUMAN:
                warning = True

            # Check for dangerous animals
            if label in DANGEROUS_ANIMALS:
                danger = True
                danger_animal = label

            # Draw bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = (0, 255, 0) if label in FARM_ANIMALS else (0, 0, 255) if label in DANGEROUS_ANIMALS else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Update database with the latest counts and statuses
    ref.update({
        "cow_count": cow_count,
        "goat_count": goat_count,
        "horse_count": horse_count,
        "warning": warning,
        "is_danger": True if danger else False,
        "danger_animal": danger_animal if danger else None,
        "last_updated": datetime.datetime.now().timestamp() * 1000  # Add timestamp in milliseconds
    })
    print(datetime.datetime.now().timestamp() * 1000)  # Print timestamp in milliseconds
    # Display live count
    info = f"Cows: {cow_count} | Goats: {goat_count} | Horses: {horse_count} | Warning: {warning} | Danger: {danger_animal if danger else 'None'}"
    cv2.putText(frame, info, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Farm Animal Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
