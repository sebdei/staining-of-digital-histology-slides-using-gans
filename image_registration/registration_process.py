import cv2
import glob
import numpy as np
import os

from pystackreg import StackReg

from joblib import Parallel, delayed
import multiprocessing


def _assure_even_dims(img):
    # sic! this code is actually valid

    if not img.shape[0] % 2 == 0:
        img = cv2.resize(img, (img.shape[0] + 1, img.shape[1]))

    if not img.shape[0] % 2 == 0:
        img = cv2.resize(img, (img.shape[0] + 1, img.shape[1]))

    return img


def _assure_square(img):
    if (img.shape[0] > img.shape[1]):
        return cv2.resize(img, (img.shape[0], img.shape[0]))
    elif (img.shape[1] > img.shape[0]):
        return cv2.resize(img, (img.shape[1], img.shape[1]))
    else:
        return img


def _align_size_crop(he, ihc):
    he = _assure_square(he)
    ihc = _assure_square(ihc)

    he = _assure_even_dims(he)
    ihc = _assure_even_dims(ihc)

    if (he.shape[0] == ihc.shape[0] and ihc.shape[1] == ihc.shape[1]):
        return he, ihc
    elif (he.shape[0] < ihc.shape[0] or ihc.shape[1] < ihc.shape[1]):
        half_diff_c = int((ihc.shape[0] - he.shape[0]) / 2)
        half_diff_r = int((ihc.shape[1] - he.shape[1]) / 2)

        ihc = ihc[half_diff_c:ihc.shape[0]-half_diff_c,
                  half_diff_r:ihc.shape[1]-half_diff_r, ]
    elif (he.shape[0] > ihc.shape[0] or ihc.shape[1] > ihc.shape[1]):
        half_diff_c = int((he.shape[0] - ihc.shape[0]) / 2)
        half_diff_r = int((he.shape[1] - ihc.shape[1]) / 2)

        he = he[half_diff_c:he.shape[0]-half_diff_c,
                half_diff_r:he.shape[1]-half_diff_r, ]

    return he, ihc


def _register_and_transform(source, target, transformation):
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    source_gray = cv2.GaussianBlur(source_gray, (5, 5), 0)
    target_gray = cv2.GaussianBlur(target_gray, (5, 5), 0)

    stack_reg = StackReg(transformation)
    transformation_matrix = stack_reg.register(source_gray, target_gray)

    # sic! StackReg returns a flipped transformation matrix. Hence the identity is flipped.
    invert_identity = np.array([
        [-1, 1, 1],
        [1, -1, 1],
        [1, 1, -1]
    ])

    transformation_matrix = transformation_matrix*invert_identity
    border_value = np.median(target[0], axis=0)

    return source, cv2.warpPerspective(target, transformation_matrix, (target.shape[0], target.shape[1]),
                                       borderValue=border_value)


def register_images(path_he, path_ihc):
    he_name = os.path.basename(path_he)
    ihc_name = os.path.basename(path_ihc)

    assert he_name[0:10] == ihc_name[0:10]

    he = cv2.imread(path_he)
    ihc = cv2.imread(path_ihc)

    he, ihc = _align_size_crop(he, ihc)

    assert he.shape[0] == ihc.shape[0]
    assert he.shape[1] == ihc.shape[1]

    he, ihc = _register_and_transform(he, ihc, StackReg.SCALED_ROTATION)
    he, ihc = _register_and_transform(he, ihc, StackReg.TRANSLATION)
    he, ihc = _register_and_transform(he, ihc, StackReg.RIGID_BODY)

    cv2.imwrite(path_he, he)
    cv2.imwrite(path_ihc, ihc)


path_hes = sorted(glob.glob('he/**'))
path_ihcs = sorted(glob.glob('ihc/**'))


num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(register_images)(
    path_he, path_ihc) for path_he, path_ihc in zip(path_hes, path_ihcs))
