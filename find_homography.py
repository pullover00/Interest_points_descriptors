!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Find the projective transformation matrix (homography) between from a source image to a target image.
"""

def find_homography_ransac(source_points: np.ndarray,
                           target_points: np.ndarray,
                           confidence: float,
                           inlier_threshold: float) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return estimated transforamtion matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the RANSAC algorithm with the
    Least-Squares algorithm to minimize the back-projection error and be robust against outliers.
    Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source (object) image [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target (scene) image [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. Euclidean distance of a point from the transformed point to be considered an inlier
    :type inlier_threshold: float

    :return: (homography, inliers, num_iterations)
        homography: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
        inliers: Is True if the point at the index is an inlier. Boolean array with shape (n,)
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: Tuple[np.ndarray, np.ndarray, int]
    """

    ######################################################
    # Initialize output parameters
    best_homography = None 
    best_inliers = np.array([], dtype=bool)
    best_error = float('inf')
    num_matches = source_points.shape[0]

    max_iterations = 1000  # High number as algorithm should be terminated before


    # Parameters
    N = source_points.shape[0]  # number of data points

    # Starting parameters
    max_iteration = 1000
    m = 4 # sample size
    e = m/N  # probability of at least one outlier-free sample

    # Computation of k
    k_numerator = np.log(1 - confidence)
    k_denominator = np.log(1 - (1 - e) ** m)
    k = k_numerator / k_denominator  # expected trials to succeed
  
    # Pre-compute some values
    target_points_homogeneous = np.vstack((target_points.T, np.ones(num_matches)))
    inlier_threshold_squared = inlier_threshold ** 2

    for iteration in range(int(max_iteration)):
        # Sample four random points
        sample_indices = np.random.choice(range(N), min_sample_size, replace=False) # choose random number
        source_sample = source_points[sample_indices] # select corresponding samples
        target_sample = target_points[sample_indices]

        # Calculate the homography matrix by using least square implementation
        homography = find_homography_leastsquares(source_sample, target_sample)

        # Apply transformation to all source points
        transformed_source_points = np.dot(homography, np.vstack((source_points.T, np.ones(N))))
        transformed_source_points = transformed_source_points[:2, :] / transformed_source_points[2, :]

        # Compute squared Euclidean distance of all transformed points to respective points on target space
        distances_squared = np.sum((transformed_source_points - target_points.T) ** 2, axis=0)

        # Identify inliers
        inliers = distances_squared < inlier_threshold_squared
        num_inliers = np.sum(inliers)

        # Calculate mean squared error
        mean_error_squared = np.mean(distances_squared)

        # Update best solution if the error is improved
        if (1 - e ** m ) ** iteration < (1 - confidence ) :
            best_homography = homography
            best_inliers = inliers
            best_error = mean_error_squared

            # Check termination based on confidence level
            if (1-(e**k)) > (1-confidence): 
                break

    # Last refinement using all inliers
    inlier_source = source_points[best_inliers] # find inliers in source
    inlier_target = target_points[best_inliers] # find inliers in target
    best_suggested_homography = find_homography_leastsquares(inlier_source, inlier_target) # last square refinement

    return best_suggested_homography, best_inliers, iteration

    ######################################################
    # Sources
    # https://www.kaggle.com/code/ravisane1/ransac-code-from-scratch
    # https://medium.com/@insight-in-plain-sight/estimating-the-homography-matrix-with-the-direct-linear-transform-dlt-ec6bbb82ee2b
    ######################################################


def find_homography_leastsquares(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Return projective transformation matrix of source_points in the target image given matching points

    Return the projective transformation matrix for homogeneous coordinates. It uses the Least-Squares algorithm to
    minimize the back-projection error with all points provided. Requires at least 4 matching points.

    :param source_points: Array of points. Each row holds one point from the source image (object image) as [x, y]
    :type source_points: np.ndarray with shape (n, 2)

    :param target_points: Array of points. Each row holds one point from the target image (scene image) as [x, y].
    :type target_points: np.ndarray with shape (n, 2)

    :return: The projective transformation matrix for homogeneous coordinates with shape (3, 3)
    :rtype: np.ndarray with shape (3, 3)
    """
    ######################################################
    # Check if we have at least 4 point correspondences
    while source_points.shape[0] < 4 or target_points.shape[0] < 4:
        continue

    # Create the coefficient matrix A
    A = np.zeros((2 * source_points.shape[0], 9)) # Initialize shape of A matrix
    for i in range(source_points.shape[0]):
        x, y = source_points[i] # convert source point tuples to x and y arrays
        xp, yp = target_points[i] # convert target point tuples to x and y arrays
        A[2*i] = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp] # even rows of A
        A[2*i+1] = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp] # uneven rows of A

    # Solve for the homography matrix using least squares
    # Single value decomposition used to extract last row of the matrix
    # Output linalg.svd: U, S, Vh
    # U: numpy.linalg.svd, S: singular values, Vh: contains the orthonormal basis vectors of the row space 
    _, _, Vh = np.linalg.svd(A)
    homography = Vh[-1].reshape(3, 3)
    
    return homography
    ######################################################
    # Sources
    # https://medium.com/@insight-in-plain-sight/estimating-the-homography-matrix-with-the-direct-linear-transform-dlt-ec6bbb82ee2b
    ######################################################
