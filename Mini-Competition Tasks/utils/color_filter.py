
import cv2 as cv
import numpy as np


def hsv_trackbars_pos(unused=None, name=None):

    """ Returns the lower and upper hsv range boundaries from the Mask Detection Trackbar """

    hue_min = cv.getTrackbarPos("Hue (Min)", name)
    hue_max = cv.getTrackbarPos("Hue (Max)", name)
    sat_min = cv.getTrackbarPos("Sat (Min)", name)
    sat_max = cv.getTrackbarPos("Sat (Max)", name)
    val_min = cv.getTrackbarPos("Val (Min)", name)
    val_max = cv.getTrackbarPos("Val (Max)", name)
    done = cv.getTrackbarPos("Done", name)

    hsv_lower_bound = np.array([hue_min, sat_min, val_min])
    hsv_upper_bound = np.array([hue_max, sat_max, val_max])

    return (hsv_lower_bound, hsv_upper_bound), done


def hsv_trackbars_create(name):

    """ Color Detection Trackbars """
    cv.namedWindow(name)
    cv.createTrackbar("Hue (Min)", name, 0, 179, hsv_trackbars_pos)
    cv.createTrackbar("Hue (Max)", name, 179, 179, hsv_trackbars_pos)
    cv.createTrackbar("Sat (Min)", name, 0, 255, hsv_trackbars_pos)
    cv.createTrackbar("Sat (Max)", name, 255, 255, hsv_trackbars_pos)
    cv.createTrackbar("Val (Min)", name, 0, 255, hsv_trackbars_pos)
    cv.createTrackbar("Val (Max)", name, 255, 255, hsv_trackbars_pos)
    cv.createTrackbar("Done", name, 0, 1, hsv_trackbars_pos)


# The following code is used to tune the color ranges for the tubes (Uncomment and run)
"""
hsv_trackbars_create()
while True:

    lower_bound, upper_bound = hsv_trackbars_pos()
    mask = cv.inRange(img_hsv, lowerb=lower_bound, upperb=upper_bound)
    img_masked = cv.bitwise_and(img_original, img_original, mask=mask)

    color_extraction_hstack = stack_images(0.8, [[img_original, img_masked], [img_hsv, mask]])

    cv.imshow("Color Extraction Stack", color_extraction_hstack)
    cv.waitKey(1)
"""