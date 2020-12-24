import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.data import stereo_motorcycle
from skimage import io
from skimage import transform

from skimage.transform import resize


image0 = io.imread("3-7-CE3B6B-T-E26EAF-3+3.he.jpg")
image1 = io.imread("3-7-CE3B6B-T-E26EAF-3+3.ihc.jpg")

image0 = resize(image0, image1.shape)

rows, cols = image0.shape[0], image0.shape[1]

src_cols = np.linspace(0, cols, 20)
src_rows = np.linspace(0, rows, 10)
src_rows, src_cols = np.meshgrid(src_rows, src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]

tform3 = transform.AffineTransform()
tform3.estimate(image0, image1)
warped = transform.warp(image0, tform3)

fig, ax = plt.subplots(nrows=2, figsize=(8, 3))

ax[1].imshow(warped, cmap=plt.cm.gray)

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()