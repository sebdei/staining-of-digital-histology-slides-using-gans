from skimage import data, io, transform
from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             plot_matches, BRIEF)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

from skimage.transform import resize


image0 = io.imread("3-7-CE3B6B-T-E26EAF-3+3.he.jpg")
image1 = io.imread("3-7-CE3B6B-T-E26EAF-3+3.ihc.jpg")

image0 = resize(image0, image1.shape)

image0 = rgb2gray(image0)
image1 = rgb2gray(image1)

keypoints1 = corner_peaks(corner_harris(image0), min_distance=5,
                          threshold_rel=0.1)
keypoints2 = corner_peaks(corner_harris(image1), min_distance=5,
                          threshold_rel=0.1)

extractor = BRIEF()

extractor.extract(image0, keypoints1)
keypoints1 = keypoints1[extractor.mask]
descriptors1 = extractor.descriptors

extractor.extract(image1, keypoints2)
keypoints2 = keypoints2[extractor.mask]
descriptors2 = extractor.descriptors


matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)


fig, ax = plt.subplots(nrows=1, ncols=1)

plt.gray()

plot_matches(ax, image0, image1, keypoints1, keypoints2, matches12)
ax.axis('off')
ax.set_title("Original Image vs. Transformed Image")

plt.show()
