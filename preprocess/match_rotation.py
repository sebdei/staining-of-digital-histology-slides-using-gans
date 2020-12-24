# All metrics are available at
# https://scikit-image.org/docs/stable/api/skimage.metrics.html

# be careful!! are, mse, nrmse are errors while ssmi and psnr are similarities therefore pick the min/max accordingly.
from skimage.metrics import mean_squared_error

import glob
import cv2
import imutils
import numpy as np


def get_similarity_with_angle(img1, img2, angle):
    img2 = imutils.rotate(img2, angle)

    return mean_squared_error(img2, img1)


def determine_max_similarity_angle(img1, img2):
    angles = [0, 90, 180, 270]

    similarities = [get_similarity_with_angle(
        img1, img2, angle) for angle in angles]

    return angles[np.argmin(similarities)], np.min(similarities)


image_paths = glob.glob("okyaz/**")
image_paths = sorted(image_paths)

assert len(image_paths) % 2 == 0

angles = []

for i in range(0, int(len(image_paths) / 2)):
    path1 = image_paths[i * 2]
    path2 = image_paths[i * 2 + 1]

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    angle, min_distance = determine_max_similarity_angle(img1, img2)

    if (angle != 0):
        print(f"{path2}: {angle}")
        image_center = tuple(np.array(img2.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        img2 = cv2.warpAffine(
            img2, rot_mat, img2.shape[1::-1], flags=cv2.INTER_LINEAR)
        cv2.imwrite(path2, img2)
