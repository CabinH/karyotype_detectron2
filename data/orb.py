import cv2
import math
import numpy as np

def orb_match(src, tar):
    detector = cv2.ORB_create(10000)
    kpts1 = detector.detect(src, None)
    kpts2 = detector.detect(tar, None)

    descriptor = cv2.xfeatures2d.BEBLID_create(0.75)

    kpts1, desc1 = descriptor.compute(src, kpts1)
    kpts2, desc2 = descriptor.compute(tar, kpts2)

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)

    return kpts1, kpts2, nn_matches

def estimate_matrix(kpts1, kpts2, nn_matches, nn_match_ratio=0.95, min_match_count=4):

    # good match
    gd_matches = [m for m, n in nn_matches if m.distance<n.distance*nn_match_ratio] 

    matrix = np.array([[0, 0, 0], [0, 0, 0]])
    match_mask = None

    if len(gd_matches) >= min_match_count:

        src_pts = np.float32([kpts1[m.queryIdx].pt for m in gd_matches]).reshape(-1, 1, 2)
        tar_pts = np.float32([kpts2[m.trainIdx].pt for m in gd_matches]).reshape(-1, 1, 2)

        matrix, ransac_mask = cv2.estimateAffinePartial2D(src_pts, tar_pts)
        match_mask = ransac_mask.ravel().tolist()

    return matrix, match_mask

def get_scale_and_degrees(matrix):
    scale, degrees = -1, 0
    if not (matrix == np.zeros((2, 3))).all():
        scale = (np.linalg.det(matrix[:2, :2]))**0.5
        degrees = math.degrees(math.asin(matrix[1, 0]/scale))
        degrees = degrees if matrix[0, 0]<0 else (180-degrees)

    return scale, degrees 

def get_confidence(match_mask):
    return np.array(match_mask).sum()