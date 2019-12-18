# Dany jest obraz kolorowy na którym widocznych jest wiele obiektów o tym samym kształcie
# i wielkości ułożonych pod różnym kątem. Znajdź rozkład orientacji obiektów i policz ile ich jest na obrazie.
# ----------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import datetime
import math

from lookPass import fill
from operationOnHistogram import drawHistogram, drawHistogramForAngle

# ----------------------------------------------------------------------------------------------------------------------

image = cv2.imread('image/kawa.jpg')
# cv2.imshow("Original Image", image)
drawHistogram(image, "Histogram for Original Image (RGB)")

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
# blackAndWhiteImage = cv2.dilate(blackAndWhiteImage, kernel, iterations=1)
cv2.imshow("Erode Binary Image", blackAndWhiteImage)

checkedPoints = np.zeros((x, y), 'uint8')
elements = np.zeros(1, 'uint8')
angles = np.zeros(38, 'uint8')

# ==================================================== FUNKCJA =========================================================
print("START")
start = datetime.datetime.now()
Ac = ~blackAndWhiteImage
result = np.zeros((x, y, 3), 'uint8')

for i in range(5, x, 20):
    for j in range(5, y, 20):
        if checkedPoints[i][j] != 255 and Ac[i][j] != 65535:
            tmpTable1 = fill(i, j, kernel, Ac, x, y, checkedPoints, elements, angles)
            result = result + tmpTable1

img = Image.fromarray(result)
img.save('wyniki/najnowszy.tif', )

# Czerwony kolor to wszystkie obiekty ktore nie spelniaja warunkow, nie zaliczaja sie
print("Znaleziono :", elements[0])
cv2.imshow("Result Image", result)
duration = datetime.datetime.now() - start
print("Dzialanie algorytmu zajelo: ", duration)
print("END")

# ======================================================================================================================
# ================================================== Histogram =========================================================
xx = [e + 1 for e in range(38)]

drawHistogramForAngle(xx, angles)
exampleResult = cv2.imread('wyniki/result.tif')
drawHistogram(exampleResult, "Histogram for proceed image(RGB)")

cv2.waitKey(0)
