import cv2
import numpy as np

img = cv2.imread('images/magnit.jpg')
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

img_copy = img.copy()

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #перевесчти в нрадацію сірого
img = cv2.GaussianBlur(img, (5, 5), 1) #розмиття

img = cv2.equalizeHist(img) #посилення контрасту

img_edges = cv2.Canny(img, 100, 100) #виявлення країв

kernel = np.ones((2, 2), np.uint8)
contours, hierarchy = cv2.findContours(img_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #

magnit_counter = 0

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        x, y, w, h = cv2.boundingRect(cnt)

        cv2.drawContours(img_copy, [cnt], -1, (0, 255, 0), 2)

        magnit_counter += 1

        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

        text_y = y - 10 if y - 10 > 20 else y + 20

        text = f'{magnit_counter}'
        cv2.putText(img_copy, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

print(f'Кількість знайдених магнітів: {magnit_counter}')

cv2.imwrite('images/magnet.jpg', img_copy)

cv2.imshow('image', img)
cv2.imshow('image2', img_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()