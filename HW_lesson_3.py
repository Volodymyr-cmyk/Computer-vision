import cv2
import numpy as np
from numpy.ma.core import filled

img = cv2.imread('images/vova.jpg')
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
print(img.shape)

cv2.rectangle(img, (110, 180), (350, 565), (55, 200, 0), 2)

cv2.putText(img, "Данилюк Володимир", (45, 590),  cv2.FONT_HERSHEY_COMPLEX, 1, (55, 200, 0), 1)

cv2.imwrite('Я.jpg', img)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()