import math
import random
import warnings

import cv2
import numpy as np
from PIL import Image
from skimage import filters
from skimage.measure import regionprops

warnings.filterwarnings("ignore")

# ======================================================================================================================

colorDefault = np.zeros(3, 'uint8')
colorDefault[0] = 0
colorDefault[1] = 0
colorDefault[2] = 0

colors = np.zeros((39, 3), 'uint16')
for i in range(39):
    colors[i][0] = 0
    colors[i][1] = random.randint(0, 255)
    colors[i][2] = random.randint(0, 255)

# Wszystkie obiekty ktore nie spelniaja warunkow beda koloru BIALEGO
R = 255
G = 255
B = 255


# ======================================================================================================================
def computeTg(A, B):
    if A[0] <= B[0] and A[1] <= B[1]:
        lengthA = B[0] - A[0]
        lengthB = B[1] - A[1]
        return -(lengthB / lengthA)
    elif A[0] >= B[0] and A[1] <= B[1]:
        lengthA = A[0] - B[0]
        lengthB = B[1] - A[1]
        return lengthB / lengthA
    elif A[0] >= B[0] and A[1] >= B[1]:
        lengthA = A[0] - B[0]
        lengthB = A[1] - B[1]
        return -(lengthB / lengthA)
    elif A[0] <= B[0] and A[1] >= B[1]:
        lengthA = B[0] - A[0]
        lengthB = A[1] - B[1]
        return lengthB / lengthA


def isclose(angle, a, b, i, angles):
    if angle >= a and angle < b:
        colorDefault[0] = colors[i, 0]
        colorDefault[1] = colors[i, 1]
        colorDefault[2] = colors[i, 2]
        angles[i + 1] = angles[i + 1] + 1
        return 1
    elif angle >= 85:
        colorDefault[0] = colors[37, 0]
        colorDefault[1] = colors[37, 1]
        colorDefault[2] = colors[37, 2]
        angles[37] = angles[37] + 1
        return 1
    elif angle <= -85:
        colorDefault[0] = colors[38, 0]
        colorDefault[1] = colors[38, 1]
        colorDefault[2] = colors[38, 2]
        angles[0] = angles[0] + 1
        return 1
    return 0


def setColorBasedOnCoefficient(angle, angles):
    flag = 0
    # Dla dodatnich
    for i in range(18):
        if isclose(angle, i * 5, (i + 1) * 5, i, angles) == 1:
            flag = 1
            break

    # Dla ujemnych
    if flag == 0:
        for i in range(18):
            if isclose(angle, (i + 1) * (-5), i * (-5), i + 18, angles) == 1:
                break


# ======================================================================================================================

# 0 to czarny a 255 to bialy
# 0 to czarny a 65535 to bialy
def fill(m, w, kernel, Ac, x, y, checkedPoints, elements, angles):
    rgbArray = np.zeros((x, y, 3), 'uint8')
    tmpTable1 = np.zeros((x, y), 'uint16')
    for i in range(x):
        for j in range(y):
            tmpTable1[i][j] = 65535

    tmpTable1[m][w] = 0
    tmpTable2 = tmpTable1

    for i in range(1200):
        tmpTable1 = cv2.erode(tmpTable1, kernel, iterations=1) | Ac
        if np.array_equal(tmpTable1, tmpTable2):
            break
        tmpTable2 = tmpTable1

    south = np.zeros(2, 'uint16')
    north = np.zeros(2, 'uint16')
    west = np.zeros(2, 'uint16')
    east = np.zeros(2, 'uint16')

    first = 0
    west[1] = y
    flag = 0

    for i in range(x):
        for j in range(y):
            if tmpTable1[i][j] == 0:
                flag = 1
                rgbArray[i][j][0] = R
                rgbArray[i][j][1] = G
                rgbArray[i][j][2] = B
                checkedPoints[i][j] = 255

                if west[1] > j:
                    west[0] = i
                    west[1] = j

                if east[1] < j:
                    east[0] = i
                    east[1] = j

                if first == 0:
                    north[0] = i
                    north[1] = j
                    first = 1

                south[0] = i
                south[1] = j

    if north[0] - 3 > 0:
        north[0] = north[0] - 3
    if west[1] - 3 > 0:
        west[1] = west[1] - 3
    if south[0] + 3 < x:
        south[0] = south[0] + 3
    if east[1] + 3 < y:
        east[1] = east[1] + 3

    # Jesli wszystkie warunki sa spelnione wykonywana jest ta czesc programu(walidacja)
    if (flag == 1) and (west[1] < east[1]) and (north[0] < south[0]):
        img = rgbArray[north[0]:south[0], west[1]:east[1], :]
        img = cv2.dilate(img, kernel, iterations=3)
        img = cv2.erode(img, kernel, iterations=3)
        count = np.count_nonzero(img)

        if count > 0:
            threshold_value = filters.threshold_otsu(img)
            labeled_foreground = (img > threshold_value).astype(int)
            properties = regionprops(labeled_foreground, img)
            center_of_mass = properties[0].centroid

            X = int(center_of_mass[0])
            Y = int(center_of_mass[1])

            img = cv2.Canny(img, 100, 200)

            # Rozmiar wycietej czesci obrazu
            dimensions = img.shape
            xx = dimensions[0]
            yy = dimensions[1]

            longestLength = 0
            point = np.zeros(2, 'uint16')

            for i in range(xx):
                for j in range(yy):
                    if img[i][j] == 255:
                        length = math.sqrt(math.pow(j - X, 2) + math.pow(i - Y, 2))
                        if length > longestLength:
                            longestLength = length
                            point[0] = j
                            point[1] = i

            longestLength = 0
            point1 = np.zeros(2, 'uint16')
            for i in range(xx):
                for j in range(yy):
                    if img[i][j] == 255:
                        length = math.sqrt(math.pow(j - point[0], 2) + math.pow(i - point[1], 2))
                        if length > longestLength:
                            longestLength = length
                            point1[0] = j
                            point1[1] = i

            srodek = np.zeros(2, 'uint16')
            srodek[0] = int((point[0] + point1[0]) / 2)
            srodek[1] = int((point[1] + point1[1]) / 2)

            shortestLength = 1000
            point2 = np.zeros(2, 'uint16')
            for i in range(xx):
                for j in range(yy):
                    if img[i][j] == 255:
                        length = math.sqrt(math.pow(j - srodek[0], 2) + math.pow(i - srodek[1], 2))
                        if length < shortestLength:
                            shortestLength = length
                            point2[0] = j
                            point2[1] = i

        if shortestLength == 0:
            shortestLength = 1

        # Nie mozna dawac punktow do metody np.polyfit ktore wskazuja na [0,0]
        if point1[0] == 0 and point1[1] == 0:
            point1[0] = 0
            point1[1] = 1
        if point[0] == 0 and point[1] == 0:
            point[0] = 0
            point[1] = 1
        if point2[0] == 0 and point2[1] == 0:
            point2[0] = 0
            point2[1] = 1

        tg = computeTg(point, point1)

        if int(longestLength) > 25 and int(longestLength) < 250:
            if longestLength / (2 * shortestLength) > 1 and longestLength / (2 * shortestLength) < 4:

                angle = int(math.degrees(math.atan(tg)))
                setColorBasedOnCoefficient(angle, angles)
                elements[0] = elements[0] + 1

                lineThickness = 1
                cv2.line(img, (point1[0], point1[1]), (point[0], point[1]), (255, 0, 0), lineThickness)
                cv2.line(img, (point2[0], point2[1]), (srodek[0], srodek[1]), (255, 0, 0), lineThickness)

                IMAGE_DESTINATION_PATH = 'results/angles/'
                img = Image.fromarray(img)
                img.save(IMAGE_DESTINATION_PATH + str(elements[0]) + '.tif')

                print("Elements number: ", elements[0])
                print("Angle for object: ", angle)
                for i in range(x):
                    for j in range(y):
                        if tmpTable1[i][j] != 65535:
                            rgbArray[i][j][0] = colorDefault[0]
                            rgbArray[i][j][1] = colorDefault[1]
                            rgbArray[i][j][2] = colorDefault[2]

    return rgbArray
