# Dany jest obraz kolorowy na którym widocznych jest wiele obiektów o tym samym kształcie
# i wielkości ułożonych pod różnym kątem. Znajdź rozkład orientacji obiektów i policz ile ich jest na obrazie.

import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------------------

image = cv2.imread('pszenica.jpg')
# cv2.imshow("Original Image", image)

grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Gray Image", grayImg)

(thresh, blackAndWhiteImage) = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Binary Image", blackAndWhiteImage)

edges = cv2.Canny(grayImg, 100, 200)
cv2.imshow("Edges Image", edges)

# Size
dimensions = edges.shape
x = dimensions[0]
y = dimensions[1]

# 0 to czarny a 255 to bialy
tmpTable1 = [[0 for i in range(x)] for j in range(y)]
tmpTable1 = np.uint8(tmpTable1)
tmpTable1 = np.array(tmpTable1)
#
tmpTable2 = tmpTable1
kernel = [[0 for i in range(3)] for j in range(3)]
kernel = np.uint8(kernel)
kernel[0][0] = 0
kernel[0][1] = 255
kernel[0][2] = 0
kernel[1][0] = 255
kernel[1][1] = 255
kernel[1][2] = 255
kernel[2][0] = 0
kernel[2][1] = 255
kernel[2][2] = 0

# edges = cv2.dilate(edges, kernel,iterations = 3)
#
# # ==================================================== FUNKCJA ===============================================
# Ac = ~edges
#
# # cv2.imshow("Inverse binary Image", Ac)
# number = np.uint8(0)
# number2 = np.uint8(255)
#
# for i in range(x):
#     for j in range(y):
#         if edges[i][j] == number2:
#             print("Total score for %s is %s" % (i, j))
#
# tmpTable1[540][89] = number2
#
# # print(blackAndWhiteImage.dtype)
# # print(number.dtype)
#
# # print(type(tmpTable1))
# # print(type(Ac))
# # print(type(edges))
# #
# # print(Ac.dtype)
# # print(tmpTable1.dtype)
#
#
# for i in range(100):
#     tmpTable1 = cv2.dilate(tmpTable1, kernel,iterations = 1) & Ac
#     # cv2.imshow("1 operations Image", tmpTable1)
#     if tmpTable1 is tmpTable2:
#         break
#     tmpTable2 = tmpTable1
#
#
#
# # cv2.imshow("1 operations Image", blackAndWhiteImage)
# # cv2.imshow("2 operations Image", tmpTable1)
# img = edges | tmpTable1
# #
# cv2.imshow("After operations Image", img)

cv2.waitKey(0)
