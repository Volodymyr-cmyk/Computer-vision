import cv2
import numpy as np

img = cv2.imread("images/sweets.jpg")
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

img_copy = img.copy()
hsv = cv2.cvtColor(img_copy, cv2.COLOR_BGR2HSV)

lower_pink = np.array([160, 100, 100])
upper_pink = np.array([180, 255, 255])

lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

lower_purple = np.array([120, 0, 0])
upper_purple = np.array([165, 255, 255])

mask1 = cv2.inRange(hsv, lower_pink, upper_pink)
mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)
mask3 = cv2.inRange(hsv, lower_purple, upper_purple)

mask = cv2.bitwise_or(mask1, mask2, mask3)

colours = [
    {
        "name": "Yellow",
        "lower": [20, 100, 100], "upper": (30, 255, 255)
    },
    {
        "name": "Pink",
        "lower": [160, 100, 100], "upper": (180, 255, 255)
    },
    {
        "name": "Purple",
        "lower": [120, 0, 0], "upper": (165, 255, 255)
    }
]

for colour in colours:
    lower = np.array(colour["lower"])
    upper = np.array(colour["upper"])
    mask = cv2.inRange(hsv, lower, upper)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.drawContours(img_copy, [cnt], -1, (255, 0, 0), 2)

            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (255, 0, 0), 2)

            text_y = y - 5 if y - 5 > 10 else y + 15

            text = colour["name"]

            size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 14, 1)

            cv2.putText(img_copy, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

cv2.imwrite("images/sweets1.jpg", img_copy)

cv2.imshow('image', img)
cv2.imshow('image2', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()