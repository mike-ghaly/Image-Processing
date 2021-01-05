import cv2 as cv
import numpy as np


def threshold_trackbars_create():
    """ Canny Threshold Trackbars """
    cv.namedWindow("Thresholds")
    cv.createTrackbar("Blur", "Thresholds", 1, 30, threshold_trackbars_pos)
    cv.createTrackbar("B", "Thresholds", 1, 50, threshold_trackbars_pos)
    cv.createTrackbar("C", "Thresholds", 1, 20, threshold_trackbars_pos)
    cv.createTrackbar("Area", "Thresholds", 1, 1000, threshold_trackbars_pos)
    cv.createTrackbar("Epsilon", "Thresholds", 1, 10, threshold_trackbars_pos)


def threshold_trackbars_pos(unused=0):
    """ Returns """
    blur = cv.getTrackbarPos("Blur", "Thresholds")
    while not blur % 2 or not blur > 1:
        blur = blur + 1

    b = cv.getTrackbarPos("B", "Thresholds")
    while not b % 2 or not b > 1:
        b = b + 1

    c = cv.getTrackbarPos("C", "Thresholds")
    area = cv.getTrackbarPos("Area", "Thresholds")
    epsilon = cv.getTrackbarPos("Epsilon", "Thresholds") / 100

    return blur, b, c, area, epsilon
