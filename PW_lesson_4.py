import os
import cv2
import time
from ultralytics import YOLO

PROJECT_DIR = os.path.dirname(__file__)
VIDEO_DIR = os.path.join(PROJECT_DIR, 'video')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')

os.makedirs(OUT_DIR, exist_ok=True)

cap = cv2.VideoCapture('video/Parking.mp4')
model = YOLO('yolov8n.pt')

CONF_THRESHOLD = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, conf=CONF_THRESHOLD, verbose=False)

    park_count = 0
    psevdo_id = 0

    PARK_CLASS_ID = 2

    a = cv2.rectangle(frame, (80, 190), (175, 240), (0, 255, 0), 1)

    for i in a:
        if a is True:
            park_count += 1

    for r in result:

        boxes = r.boxes
        if boxes is None:
            continue

        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])


            if cls == PARK_CLASS_ID:
                park_count += 1
                psevdo_id += 1

                cv2.rectangle(frame, (80, 190), (175, 240), (0, 255, 0), 1)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

                label = f"id:{psevdo_id} conf:{round(conf, 2)}"
                cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Parking lot count: {park_count}', (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (255, 0, 0), 1)
    cv2.imshow('YOLO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
