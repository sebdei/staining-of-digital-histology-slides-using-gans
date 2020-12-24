from skimage.feature import register_translation
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import io
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
from skimage.color import rgb2gray
from skimage.transform import AffineTransform, EuclideanTransform, warp


image0_original = io.imread("3-7-CE3B6B-T-E26EAF-3+3.he.jpg")
image1_original = io.imread("3-7-CE3B6B-T-E26EAF-3+3.ihc.jpg")

image0 = rgb2gray(image0_original)
image1 = rgb2gray(image1_original)

shift, error, diffphase = phase_cross_correlation(image1, image0)

print(shift)

transform = EuclideanTransform(rotation=0, translation=shift)
shifted = warp(image1_original, transform)

io.imsave("register_translation.jpg", shifted)

fig, ax0 = plt.subplots(1, 1)

ax0.imshow(shifted)
ax0.set_title("Unregistered sequence")
ax0.set_axis_off()

plt.show()
