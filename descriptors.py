#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Descriptor function

Author: FILL IN
MatrNr: FILL IN
"""
from typing import List, Tuple

import numpy as np
import cv2

def compute_descriptors(img: np.ndarray,
                        keypoints: List[cv2.KeyPoint],
                        patch_size: int) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
    """ Calculate a descriptor on patches of the image, centred on the locations of the KeyPoints.

    Calculate a descriptor vector for each keypoint in the list. KeyPoints that are too close to the border to include
    the whole patch are filtered out. The descriptors are returned as a k x m matrix with k being the number of filtered
    KeyPoints and m being the length of a descriptor vector (patch_size**2). The descriptor at row i of
    the descriptors array is the descriptor for the KeyPoint filtered_keypoint[i].

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param keypoints: List of keypoints at which to compute the descriptors
    :type keypoints: List[cv2.KeyPoint]

    :param patch_size: Value defining the width and height of the patch around each keypoint to calculate descriptor.
    :type patch_size: int

    :return: (filtered_keypoints, descriptors):
        filtered_keypoints: List of the filtered keypoints.
            Locations too close to the image boundary to cut out the image patch should not be contained.
        descriptors: k x m matrix containing the patch descriptors.
            Each row vector stores the descriptor vector of the respective corner.
            with k being the number of descriptors and m being the length of a descriptor (usually patch_size**2).
            The descriptor at row i belongs to the KeyPoint at filtered_keypoints[i]
    :rtype: (List[cv2.KeyPoint], np.ndarray)
    """
    ######################################################
    # Write your own code here
    descriptors = []
    filtered_keypoints = []

      # Keypoints to coordinates
    for keypoint in keypoints:
        x, y = keypoint.pt
      
        # Find borders of the patch
        x1 = int(x - patch_size // 2)
        x2 = int(x + patch_size // 2)
        y1 = int(y - patch_size // 2)
        y2 = int(y + patch_size // 2)

        # Check if patches lie within image borders
        if x1 < 0 or x2 >= img.shape[1] or y1 < 0 or y2 >= img.shape[0]:
            continue

        # Get patch from image
        patch = img[y1:y2, x1:x2]
        patch_norm = (patch - np.mean(patch)) / np.std(patch)  # Normalize patch

        # Generate descriptor
        descriptor = patch_norm.flatten()  # Flatten patch
        descriptors.append(descriptor)  # Append descriptor to list
        filtered_keypoints.append(keypoint)

    return filtered_keypoints, np.array(descriptors, dtype=np.float32)

