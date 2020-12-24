import numpy as np
from matplotlib import pyplot as plt

from skimage import data
from skimage import io
from skimage.util import img_as_float
from skimage.feature import (corner_harris, corner_subpix, corner_peaks,
                             plot_matches)
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac

from skimage.transform import resize


image0 = io.imread("3-7-CE3B6B-T-E26EAF-3+3.he.jpg")
image0 = resize(image0, (4880, 4880))
image0 = rescale_intensity(image0)
image1 = io.imread("3-7-CE3B6B-T-E26EAF-3+3.ihc.jpg")
image1 = resize(image1, (4880, 4880))
image1 = rescale_intensity(image1)

image0_grey = rgb2gray(image0)
image1_grey = rgb2gray(image1)


coords_orig = corner_peaks(corner_harris(image0_grey), threshold_rel=0.001, min_distance=5)
coords_warped = corner_peaks(corner_harris(image1_grey), threshold_rel=0.001, min_distance=5)

coords_orig_subpix = corner_subpix(image0_grey, coords_orig, window_size=8)
coords_warped_subpix = corner_subpix(image1_grey, coords_warped, window_size=8)


def gaussian_weights(window_ext, sigma=1):
    y, x = np.mgrid[-window_ext:window_ext+1, -window_ext:window_ext+1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
    g /= 2 * np.pi * sigma * sigma

    return g


def match_corner(coord, window_ext=4):
    r, c = np.round(coord).astype(np.intp)
    window_orig = image0[r-window_ext:r+window_ext+1, c-window_ext:c+window_ext+1, :]


    # weight pixels depending on distance to center pixel
    weights = gaussian_weights(window_ext, 3)
    weights = np.dstack((weights, weights, weights))

    # compute sum of squared differences to all corners in warped image
    SSDs = []
    for cr, cc in coords_warped:
        window_warped = image1[cr-window_ext:cr+window_ext+1, cc-window_ext:cc+window_ext+1, :]

        if (window_orig.shape[0] != 0):
            SSD = np.sum(weights * (window_orig - window_warped)**2)
            SSDs.append(SSD)

    if (len(SSDs) == 0):
        return None

    # use corner with minimum SSD as correspondence
    min_idx = np.argmin(SSDs)

    return coords_warped_subpix[min_idx]


src = []
dst = []
for coord in coords_orig_subpix:
    matched_corner = match_corner(coord)
    if (matched_corner is not None):
        src.append(coord)
        dst.append(matched_corner)

src = np.array(src)
src = src[~np.isnan(src)]
dst = np.array(dst)
dst = dst[~np.isnan(dst)]

# model = AffineTransform()
# model.estimate(src, dst)
# print("Affine transform:")
# print(f"Scale: ({model.scale[0]:.4f}, {model.scale[1]:.4f}), "
#       f"Translation: ({model.translation[0]:.4f}, "
#       f"{model.translation[1]:.4f}), "
#       f"Rotation: {model.rotation:.4f}")

model_robust, inliers = ransac((src, dst), AffineTransform, min_samples=3,
                               residual_threshold=2, max_trials=100)
outliers = inliers == False

print(f"Scale: ({model_robust.scale[0]:.4f}, {model_robust.scale[1]:.4f}), "
      f"Translation: ({model_robust.translation[0]:.4f}, "
      f"{model_robust.translation[1]:.4f}), "
      f"Rotation: {model_robust.rotation:.4f}")