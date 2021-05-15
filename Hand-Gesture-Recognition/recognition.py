import cv2 as cv
import numpy as np
import math
import pyautogui

from color_filter import *
from images_stack import *

# Pass camera source as 0 for built-in webcam, 1 for external camera
capture = cv.VideoCapture(0)

hsv_trackbars_create("Color Filter")
while True:

    # Obtain frame from camera capture
    _, frame = capture.read()

    # Horizontally flip the frame
    frame = cv.flip(frame, 1)

    # Define region of interest
    square_side_length = 300
    upper_left = 300
    lower_left = 100
    upper_right = upper_left + square_side_length
    lower_right = lower_left + square_side_length
    roi = frame[lower_left:lower_right, upper_left:upper_right]
    cv.rectangle(frame, (upper_left, lower_left), (upper_right, lower_right), (252, 0, 0), 0)

    # convert roi to HSV and filter hand color
    roi_hsv = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

    lower_bound, upper_bound = hsv_trackbars_pos(name="Color Filter")[0]
    mask = cv.inRange(roi_hsv, lowerb=lower_bound, upperb=upper_bound)
    roi_masked = cv.bitwise_and(roi, roi, mask=mask)

    # Morphological operations
    kernel_9 = cv.getStructuringElement(cv.MORPH_RECT, (9, 9))
    kernel_7 = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))

    # Remove Noise
    mask_opened = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel_9)
    # Close Gaps
    mask_closed = cv.morphologyEx(mask_opened, cv.MORPH_CLOSE, kernel_7)
    # Blur
    mask_blurred = cv.medianBlur(mask_closed, 5)

    # Compute the contours
    contours, _ = cv.findContours(mask_blurred, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find contour of max area (hand)
    try:
        hand_contour = max(contours, key=lambda x: cv.contourArea(x))

        # approx the contour a little
        epsilon = 0.0005 * cv.arcLength(hand_contour, True)
        approx = cv.approxPolyDP(hand_contour, epsilon, True)

        # make convex hull around hand
        hull = cv.convexHull(hand_contour)

        # define area of hull and area of hand
        area_hull = cv.contourArea(hull)
        area_hand_contour = cv.contourArea(hand_contour)

        # find the percentage of area not covered by hand in convex hull
        area_ratio = ((area_hull - area_hand_contour) / area_hand_contour) * 100

        # find the defects in convex hull with respect to hand
        hull = cv.convexHull(approx, returnPoints=False)
        defects = cv.convexityDefects(approx, hull)

        # finding number of defects due to fingers (= l)
        l = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)

            # find length of all sides of triangle
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            s = (a + b + c) / 2
            ar = math.sqrt(s * (s - a) * (s - b) * (s - c))

            # distance between point and convex hull
            d = (2 * ar) / a

            # apply cosine rule here
            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

            # ignore angles > 90 and ignore points very close to convex hull (generally noise induced points)
            if angle <= 90 and d > 30:
                l += 1
                cv.circle(roi, far, 3, [0, 0, 255], -1)

            # draw lines around hand
            cv.line(roi, start, end, [255, 0, 0], 2)

        # minimum one defect for hand
        l += 1
        (x, y), (MA, ma), angle = cv.fitEllipse(approx)
        print(f"Defects: {l}\tAngle: {angle}")

        """ Chrome Dino Game """
        if l == 1:
            pyautogui.press('space')


    except:
        pass

    color_extraction_stack = stack_images(0.8, [[roi, mask], [roi_hsv, mask_blurred]])
    cv.imshow("Color Extraction Stack", color_extraction_stack)

    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()
capture.release()
