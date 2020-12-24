import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.data import stereo_motorcycle
from skimage.transform import PolynomialTransform, warp, warp_polar
from skimage.registration import optical_flow_tvl1
from skimage import io

from skimage.transform import resize


# --- Load the sequence
image0_color = io.imread("3-7-CE3B6B-T-E26EAF-3+3.he.jpg")
image1_color = io.imread("3-7-CE3B6B-T-E26EAF-3+3.ihc.jpg")

image0_color = resize(image0_color, image1_color.shape)

# --- Convert the images to gray level: color is not supported.
image0 = rgb2gray(image0_color)
image1 = rgb2gray(image1_color)

# --- Compute the optical flow
v, u = optical_flow_tvl1(image0, image1)

# # --- Use the estimated optical flow for registration

nr, nc = image0.shape

row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

transform = np.array([row_coords + v, col_coords + u])
image1_warp = warp(image1, transform, mode='nearest')

# build an RGB image with the unregistered sequence
seq_im = np.zeros((nr, nc, 3))
seq_im[..., 0] = image1
seq_im[..., 1] = image0
seq_im[..., 2] = image0

# build an RGB image with the registered sequence
reg_im = np.zeros((nr, nc, 3))
reg_im[..., 0] = image1_warp
reg_im[..., 1] = image0
reg_im[..., 2] = image0

# build an RGB image with the registered sequence
target_im = np.zeros((nr, nc, 3))
target_im[..., 0] = image0
target_im[..., 1] = image0
target_im[..., 2] = image0

# 
result = np.zeros((nr, nc, 3))
result[..., 0] = image1_warp
result[..., 1] = image1_warp
result[..., 2] = image1_warp

# --- Show the result

fig, (ax0, ax1, ax2, ax3) = plt.subplots(2, 2, figsize=(5, 10))

ax0.imshow(seq_im)
ax0.set_title("Unregistered sequence")
ax0.set_axis_off()

ax1.imshow(reg_im)
ax1.set_title("Registered sequence")
ax1.set_axis_off()

ax2.imshow(target_im)
ax2.set_title("Target")
ax2.set_axis_off()

ax3.imshow(result)
ax3.set_title("Result grey")
ax3.set_axis_off()

fig.tight_layout()
plt.show()
