import glob
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt


def binarize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_TOZERO)

    return thresh


img1 = cv2.imread("./images/6-4-46C0AD-T-18248A-3+3.ihc.jpg")
grey = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img2 = cv2.imread("./images/6-4-46C0AD-T-18248A-3+3.ihc.jpg")

# edges = binarize(img1)

img1_ = cv2.GaussianBlur(grey, (3, 3), cv2.BORDER_DEFAULT)
edges = cv2.Canny(img1_, 100, 200)

contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))

cv2.drawContours(img1, contours, -1, (0, 255, 0), 3)

cv2.imwrite('Contour.jpg', img1)
