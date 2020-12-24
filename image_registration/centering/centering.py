import glob
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread("./images/6-4-46C0AD-T-18248A-3+3.he.jpg")
img2 = cv2.imread("./images/6-4-46C0AD-T-18248A-3+3.ihc.jpg")


def findCenter(img, i):
    print(img.shape, img.dtype)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(
        gray, 150, 255, cv2.THRESH_TOZERO)

    cv2.imwrite(f"tresh{i}.png", threshed)

    #cv2.imshow("threshed", threshed);cv2.waitKey();cv2.destroyAllWindows()
    #_, cnts, hierarchy = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(threshed, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(cnts[0])
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return (cX, cY)


pt1 = findCenter(img1, 1)
pt2 = findCenter(img2, 2)

# (2) Calc offset
dx = pt1[0] - pt2[0]
dy = pt1[1] - pt2[1]

# (3) do slice-op `paste`
h, w = img2.shape[:2]

dst = img1.copy()
dst[dy:dy+h, dx:dx+w] = img2

cv2.imwrite("images/res.png", dst)


# cv2.imwrite('i.jpg', img)
