import glob
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt


def binarize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_TOZERO)

    return thresh


def crop(img, boundaries):
    minx, miny, maxx, maxy = boundaries
    minimum = min(minx, miny)
    maximum = max(maxx, maxy)
    # minimum = int((minx + miny) / 2)
    # maximum = int((maxx + maxy) / 2)

    return img[minimum:maximum, minimum:maximum]


img_original = cv2.imread("./images/6-4-46C0AD-T-18248A-3+3.he.jpg")

img_original = crop(img_original, (200, 200,
                                   img_original.shape[0] - 200, img_original.shape[1] - 200))

img = cv2.GaussianBlur(img_original, (5, 5), 0)
img = cv2.medianBlur(img, 151)
img = binarize(img)

cv2.imwrite('bin.jpg', img)

dilated = cv2.morphologyEx(
    img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))
contours, _ = cv2.findContours(
    dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


contours = [c for c in contours if cv2.contourArea(c) < 100000]
areas = [cv2.contourArea(c) for c in contours]

print(max(areas))


def find_boundaries(img, contours):
    # margin is the minimum distance from the edges of the image, as a fraction
    ih, iw = img.shape[:2]
    minx = iw
    miny = ih
    maxx = 0
    maxy = 0

    for cc in contours:
        x, y, w, h = cv2.boundingRect(cc)
        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x + w > maxx:
            maxx = x + w
        if y + h > maxy:
            maxy = y + h

    return (minx, miny, maxx, maxy)


boundaries = find_boundaries(img, contours)
img = crop(img_original, boundaries)

cv2.imwrite('crop_he.jpg', img)
