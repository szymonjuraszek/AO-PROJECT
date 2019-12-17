# Dany jest obraz kolorowy na którym widocznych jest wiele obiektów o tym samym kształcie
# i wielkości ułożonych pod różnym kątem. Znajdź rozkład orientacji obiektów i policz ile ich jest na obrazie.
# ----------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
from PIL import Image
from lookPass import fill
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------

image = cv2.imread('image/13.jpg')
# cv2.imshow("Original Image", image)

grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Image", grayImg)

(thresh, blackAndWhiteImage) = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
dimensions = blackAndWhiteImage.shape
x = dimensions[0]
y = dimensions[1]
blackAndWhiteImage = np.uint16(blackAndWhiteImage)

for i in range(x):
    for j in range(y):
        if blackAndWhiteImage[i][j] == 255:
            blackAndWhiteImage[i][j] = 65535
# cv2.imshow("Binary Image", blackAndWhiteImage)

kernel = [[0 for i in range(3)] for j in range(3)]
kernel = np.uint8(kernel)
kernel[0][0] = 0
kernel[0][1] = 65535
kernel[0][2] = 0
kernel[1][0] = 65535
kernel[1][1] = 65535
kernel[1][2] = 65535
kernel[2][0] = 0
kernel[2][1] = 65535
kernel[2][2] = 0

blackAndWhiteImage = cv2.erode(blackAndWhiteImage, kernel, iterations=1)
cv2.imshow("Erode Binary Image", blackAndWhiteImage)

# ==================================================== FUNKCJA =========================================================
Ac = ~blackAndWhiteImage
result = np.zeros((x, y, 3), 'uint8')

for i in range(5, x, 20):
    for j in range(5, y, 20):
        tmpTable1 = fill(i, j, kernel, Ac, x, y)
        result = result + tmpTable1

img = Image.fromarray(result)
img.save('Result.jpeg')

# Czerwony kolor to wszystkie obiekty ktore nie spelniaja warunkow, nie zaliczaja sie
cv2.imshow("Result Image", result)

# ======================================================================================================================
# ================================================== Histogram =========================================================

img123 = cv2.imread('image/Result.jpeg')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img123], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

cv2.waitKey(0)
