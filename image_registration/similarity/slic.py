
from skimage.segmentation import slic, mark_boundaries
from skimage import img_as_float
from skimage import io
from skimage.color import rgb2gray
import numpy as np
import glob
import os
import cv2


def get_slic_segments(image):
    segments = slic(image, n_segments=100, sigma=5)

    white_img = np.zeros(image.shape, dtype=np.uint8)
    white_img.fill(0)

    result = mark_boundaries(white_img, segments)

    return result


image_paths = [glob.glob("images/*")]
image_paths = sorted(image_paths)

print(len(image_paths[0]))

assert len(image_paths[0]) % 2 == 0

for path in image_paths[0]:
    img = io.imread(path)
    # img = rgb2gray(img)
    img = img_as_float(img)

    slic_image = get_slic_segments(img)

    filename = os.path.basename(path)
    io.imsave(f"./results/slic/{filename}", slic_image)
