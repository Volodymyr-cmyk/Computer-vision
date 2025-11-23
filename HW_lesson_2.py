import cv2
import numpy as np

# Власне фото
# image = cv2.imread('images/vova.jpg')
# image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# image = cv2.Canny(image, 50, 50)
# kernel = np.ones((3, 3), np.uint8)
# image = cv2.dilate(image, kernel, iterations=1)
# image = cv2.erode(image, kernel, iterations=1)
# cv2.imwrite('images/vova1.jpg', image)


# Електронна пошта
image = cv2.imread('images/gmail.jpg')
image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.Canny(image, 100, 100)
cv2.imwrite('images/vova1.jpg', image)
kernel = np.ones((2, 2), np.uint8)
image = cv2.dilate(image, kernel, iterations=1)
image = cv2.erode(image, kernel, iterations=1)
cv2.imwrite('images/gmail1.jpg', image)

cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()