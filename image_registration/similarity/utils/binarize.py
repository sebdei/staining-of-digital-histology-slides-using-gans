import cv2


def binarize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 210, 255, cv2.THRESH_TOZERO)

    return thresh
