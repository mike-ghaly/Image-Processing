
import cv2 as cv
import numpy as np


def hsv_trackbars_pos(unused=0):

    """ Returns the lower and upper hsv range boundaries from the Mask Detection Trackbar """

    hue_min = cv.getTrackbarPos("Hue (Min)", "Mask Detection")
    hue_max = cv.getTrackbarPos("Hue (Max)", "Mask Detection")
    sat_min = cv.getTrackbarPos("Sat (Min)", "Mask Detection")
    sat_max = cv.getTrackbarPos("Sat (Max)", "Mask Detection")
    val_min = cv.getTrackbarPos("Val (Min)", "Mask Detection")
    val_max = cv.getTrackbarPos("Val (Max)", "Mask Detection")

    hsv_lower_bound = np.array([hue_min, sat_min, val_min])
    hsv_upper_bound = np.array([hue_max, sat_max, val_max])

    return hsv_lower_bound, hsv_upper_bound


def hsv_trackbars_print_pos(unused=0):
    trackbars_pos = hsv_trackbars_pos()
    print(f"Lower Bound: {list(trackbars_pos[0])}")
    print(f"Upper Bound: {list(trackbars_pos[1])}")


def hsv_trackbars_create():

    """ Mask Detections Trackbars """
    cv.namedWindow("Mask Detection")
    cv.createTrackbar("Hue (Min)", "Mask Detection", 0, 179, hsv_trackbars_pos)
    cv.createTrackbar("Hue (Max)", "Mask Detection", 179, 179, hsv_trackbars_pos)
    cv.createTrackbar("Sat (Min)", "Mask Detection", 0, 255, hsv_trackbars_pos)
    cv.createTrackbar("Sat (Max)", "Mask Detection", 255, 255, hsv_trackbars_pos)
    cv.createTrackbar("Val (Min)", "Mask Detection", 0, 255, hsv_trackbars_pos)
    cv.createTrackbar("Val (Max)", "Mask Detection", 255, 255, hsv_trackbars_pos)
    cv.createTrackbar("Print", "Mask Detection", 0, 1, hsv_trackbars_print_pos)


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