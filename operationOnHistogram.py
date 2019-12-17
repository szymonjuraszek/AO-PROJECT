import matplotlib.pyplot as plt
import cv2


def drawHistogram(image, description):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        histr = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])
    plt.title(description)
    plt.xlabel("Possible value for Color RGB type [0 to 255]")
    plt.ylabel("Number of occurrences")
    plt.show()

def drawHistogramForAngle(x,y):
    plt.bar(x, y, align='center')  # A bar chart
    plt.xlabel('to 0 -> 18 (minus degree) 19 -> 38 (positive)')
    plt.ylabel('Frequency')
    plt.show()