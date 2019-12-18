# Dany jest obraz kolorowy na którym widocznych jest wiele obiektów o tym samym kształcie
# i wielkości ułożonych pod różnym kątem. Znajdź rozkład orientacji obiektów i policz ile ich jest na obrazie.
# ----------------------------------------------------------------------------------------------------------------------

import datetime
import os, shutil

import cv2
import numpy as np
from PIL import Image

from lookPass import fill
from operationOnHistogram import drawHistogram, drawHistogramForAngle


def removeAnglesIamgeFromResult():
    PATH_TO_FOLDER = 'results/angles'
    for filename in os.listdir(PATH_TO_FOLDER):
        file_path = os.path.join(PATH_TO_FOLDER, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


# ----------------------------------------------------------------------------------------------------------------------
removeAnglesIamgeFromResult()
IMAGE_SOURCE_PATH = 'image/jpg/'
IMAGE_NAME = input("Enter the file name with the extension:    ")

thresholdForBinaryImage = input("Enter threshold for Binary conversion from 0 to 255:    ")
thresholdForBinaryImage = int(thresholdForBinaryImage)
if thresholdForBinaryImage < 0 or thresholdForBinaryImage > 255:
    exit(-1)

image = cv2.imread(IMAGE_SOURCE_PATH + IMAGE_NAME)
# cv2.imshow("Original Image", image)
drawHistogram(image, "Histogram for Original Image (RGB)")

grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Image", grayImg)

(thresh, blackAndWhiteImage) = cv2.threshold(grayImg, thresholdForBinaryImage, 255, cv2.THRESH_BINARY)

# If program cant find any object you can try use this operation because somethimes image can be convert reversed
# For example image 'kawa.jpg' you have to use it
# ===============================================================
# blackAndWhiteImage = ~blackAndWhiteImage
# ===============================================================

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

IMAGE_DESTINATION_PATH = 'results/'
img = Image.fromarray(result)
img.save(IMAGE_DESTINATION_PATH + IMAGE_NAME, )

# Bialy kolor to wszystkie obiekty ktore nie spelniaja warunkow, nie zaliczaja sie
print("Znaleziono :", elements[0])
cv2.imshow("Result Image", result)
duration = datetime.datetime.now() - start
print("Dzialanie algorytmu zajelo: ", duration)

# ======================================================================================================================
# ================================================== Histogram =========================================================
xx = [e + 1 for e in range(38)]

drawHistogramForAngle(xx, angles)
drawHistogram(result, "Histogram for proceed image(RGB)")
print("END")

cv2.waitKey(0)
