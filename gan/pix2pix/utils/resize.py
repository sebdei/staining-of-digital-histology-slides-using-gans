import cv2


def align_sizes(path_1, path_2):
    img1 = cv2.imread(path_1)
    img2 = cv2.imread(path_2)

    img1 = _assure_square(img1)
    img2 = _assure_square(img2)

    if (img1 is None or img2 is None or (img1.shape[0] == img2.shape[0] and img2.shape[1] == img2.shape[1])):
        return
    elif (img1.shape[0] > img2.shape[0]):
        img1 = cv2.resize(img1, (img2.shape[0], img2.shape[1]))
        cv2.imwrite(path_1, img1)
    else:
        img2 = cv2.resize(img2, (img1.shape[0], img1.shape[1]))
        cv2.imwrite(path_2, img2)


def _assure_square(img):
    if (img.shape[0] > img.shape[1]):
        return cv2.resize(img, (img.shape[0], img.shape[0]))
    elif (img.shape[1] > img.shape[0]):
        return cv2.resize(img, (img.shape[1], img.shape[1]))
    else:
        return img
