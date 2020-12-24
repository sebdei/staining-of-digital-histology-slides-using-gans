import cv2
import matplotlib.pyplot as plt
import numpy as np


moving = cv2.imread("3-6-49ED52-T-A719A8-3+3.ihc.jpg")
target = cv2.imread("3-6-49ED52-T-A719A8-3+3.he.jpg")

img1 = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

height, width = img2.shape

orb_detector = cv2.ORB_create(1000)

keypoints_1, d1 = orb_detector.detectAndCompute(img1, None)
keypoints_2, d2 = orb_detector.detectAndCompute(img2, None)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = matcher.match(d1, d2)
matches.sort(key=lambda x: x.distance)

matches = matches[:int(len(matches)*90)]
no_of_matches = len(matches)

p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))

for i in range(len(matches)):
    p1[i, :] = keypoints_1[matches[i].queryIdx].pt
    p2[i, :] = keypoints_2[matches[i].trainIdx].pt

homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

transformed_img = cv2.warpPerspective(moving, homography, (width, height))

img3 = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2,
                       matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3)
cv2.imwrite('output_orb_matches.jpg', img3)

cv2.imwrite('output_orb.jpg', transformed_img)
