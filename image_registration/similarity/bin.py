import cv2
from utils.binarize import binarize

img = cv2.imread("images/3-6-49ED52-T-A719A8-3+3.he.jpg")
cv2.imwrite("grey.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
