import os

import cv2
import csv

net = cv2.dnn.readNetFromCaffe("data/MobileNet/mobilenet_deploy.prototxt", "data/MobileNet/mobilenet.caffemodel")
classes = []

with open('data/MobileNet/synset.txt', 'r', encoding = 'utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        parts = line.split(' ', 1)
        name = parts[1] if len(parts) > 1 else parts[0]
        classes.append(name)

image_folder = 'images/MobileNet'

format = ('.jpg', '.png')

images = [
    f for f in os.listdir(image_folder)
    if f.endswith(format)
]

detected = []
for file in images:
    image = cv2.imread(os.path.join(image_folder, file))

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (224, 224)), 1.0 / 127.5, (224, 224), (127.5, 127.5, 127.5))

    net.setInput(blob)
    preds = net.forward() #preds - імовірність

    index = preds[0].argmax()

    label = classes[index] if index < len(classes) else "unknown"
    conf = float(preds[0][index].item()) * 100

    detected.append((file, label, conf))

    print(f'Клас: {label}')
    print(f'Ймовірність: {round(conf, 2)}%')

    with open("result.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Файл", "Клас", "Ймовірність"])
        for file, label, conf in detected:
            writer.writerow([file, label, round(conf, 2)])

    text = label + ": " + str(int(conf)) + "%"
    cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow('result', image)
    cv2.waitKey(0)
cv2.destroyAllWindows()