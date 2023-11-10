#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Corner detection with the Harris corner detector

Author: FILL IN
MatrNr: FILL IN
"""
import numpy as np
import cv2
import math # kernel computation taken from exercise 1

from typing import List

from helper_functions import non_max


def harris_corner(img: np.ndarray,
                  sigma1: float,
                  sigma2: float,
                  k: float,
                  threshold: float) -> List[cv2.KeyPoint]:
    """ Detect corners using the Harris corner detector

    In this function, corners in a grayscale image are detected using the Harris corner detector.
    They are returned in a list of OpenCV KeyPoints (https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html).
    Each KeyPoint includes the attributes, pt (position), size, angle, response. The attributes size and angle are not
    relevant for the Harris corner detector and can be set to an arbitrary value. The response is the result of the
    Harris corner formula.

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma1: Sigma for the first Gaussian filtering
    :type sigma1: float

    :param sigma2: Sigma for the second Gaussian filtering
    :type sigma2: float

    :param k: Coefficient for harris formula
    :type k: float

    :param threshold: corner threshold
    :type threshold: float

    :return: keypoints:
        corners: List of cv2.KeyPoints containing all detected corners after thresholding and non-maxima suppression.
            Each keypoint has the attribute pt[x, y], size, angle, response.
                pt: The x, y position of the detected corner in the OpenCV coordinate convention.
                size: The size of the relevant region around the keypoint. Not relevant for Harris and is set to 1.
                angle: The direction of the gradient in degree. Relative to image coordinate system (clockwise).
                response: Result of the Harris corner formula R = det(M) - k*trace(M)**2
    :rtype: List[cv2.KeyPoint]

    """
    
    ######################################################
    
    # Compute gaussian for sigma1
    kernel_width = 2*math.ceil(3*sigma1)+1 # taken from exercise 1
    kernel = cv2.getGaussianKernel(kernel_width, sigma1) # Produces an array with ( kernel_width x 1)
    gauss_1 = np.outer(kernel, kernel.transpose()) # Produces an Gaussian kernel of shape ( kernel_width x kernel_width)

    # Compute gradients
    gauss_derx = cv2.filter2D(gauss_1, ddepth=-1, kernel=np.array([[-1, 0, 1]])) # filter2D to handle borders

    gauss_dery = cv2.filter2D(gauss_1, ddepth=-1, kernel=np.array([[-1], [0], [1]]))

    # Compute horizontal and vertical derivatives of image
    I_x = cv2.filter2D(img, ddepth=-1, kernel=gauss_derx)
    I_y = cv2.filter2D(img, ddepth=-1, kernel=gauss_dery)

    # Calculate products
    I_x2 = I_x * I_x
    I_y2 = I_y * I_y
    I_xy = I_x * I_y

    # Compute gaussian for sigma2
    kernel_width = 2*math.ceil(3*sigma2)+1 # taken from exercise 1
    # Weighted products
    wI_x2 = cv2.GaussianBlur(I_x2, (kernel_width, kernel_width), sigma2)
    wI_y2 = cv2.GaussianBlur(I_y2, (kernel_width, kernel_width), sigma2)
    wI_xy = cv2.GaussianBlur(I_xy, (kernel_width, kernel_width), sigma2)

    # Harris function
    M = wI_x2 * wI_y2 - wI_xy * wI_xy
    trace = wI_x2 + wI_y2
    R = M - k * (trace ** 2) # Harris response
     
    # Normalize harris 
    R_norm = cv2.normalize(R, R, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    # Apply non max suppression
    R_nonmax = non_max(R_norm)
    
    # Apply threshold
    keypoints = []  # Initialize keypoints
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
    # Qualified as keypoint when thresholding and nonmax su
            if R_norm[y, x] > threshold and R_nonmax[y, x] == True:
                keypoints.append(cv2.KeyPoint(x, y, 1, 0, float(R_norm[y, x])))

    ######################################################
    return keypoints
    ######################################################
    # Sources:
    # https://www.geekering.com/programming-languages/python/brunorsilva/harris-corner-detector-python/
    # https://www.kaggle.com/code/dasmehdixtr/harris-corner-detector-example-from-scratch
    # https://github.com/jimmy133719/Harris-corner-detection/blob/master/main.py
    ######################################################
