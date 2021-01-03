
import numpy as np
import cv2 as cv
from color_filter import *
from images_stack import *


def distance_to_object(object_width_in_cm, camera_focal_length_in_mm, width_type):

    """
        The function computes the distance to an object using the pinhole projection formula:

                                    10 * Camera's Lens Focal Length (mm) * Object's real Width (cm)
        Distance to object (cm) = -------------------------------------------------------------------
                                                    Object's Width (pixels)


        Inputs:
            - object_width_in_cm:           (float)
            - camera_focal_length_in_mm:    (float)
            - width_type:                   (string)
                    "smaller" if the object's width is the smaller dimension between the object's width and height
                    "bigger" if the object's width is the bigger dimension between the object's width and height

        Returns:
            - None
    """

    # The trackbars are used initially to obtain the object's color range
    hsv_trackbars_create()
    while True:
        # ________________________________________________________________ Capture frame-by-frame
        ret, frame = capture.read()

        # ________________________________________________________________ Operations on the frame:
        frame = cv.resize(frame, (400, 300))
        frame = cv.GaussianBlur(frame, (5, 5), 1)
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        # ________________________________________________________________ Color Filtering
        lower_bound, upper_bound = hsv_trackbars_pos()
        # lower_bound = np.array([15, 180, 120])
        # upper_bound = np.array([20, 240, 255])
        mask = cv.inRange(frame_hsv, lowerb=lower_bound, upperb=upper_bound)
        frame_masked = cv.bitwise_and(frame, frame, mask=mask)

        # ________________________________________________________________ Finding the object's contour
        contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        desired_contour = []
        for contour_index in range(len(contours)):
            # Check if the contour is relevant
            if cv.arcLength(contours[contour_index], True) > 100:
                desired_contour.append(contours[contour_index])

        if len(desired_contour) == 1:
            rect = cv.minAreaRect(desired_contour[0])
            object_size = rect[1]
            # ____________________________________________________________ Drawing the bounding rectangle
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(frame, [box], 0, (50, 50, 50), 2)

            if width_type == "smaller":
                object_width_in_pixels = min(object_size)
            elif width_type == "bigger":
                object_width_in_pixels = max(object_size)
            else:
                # Will cause ZeroDivisionError
                object_width_in_pixels = 0

            distance = (camera_focal_length_in_mm * object_width_in_cm * 10) / object_width_in_pixels
            distance = round(distance, 2)
            cv.putText(frame, f"Distance to Object: {distance}cm", (20, 40), cv.QT_FONT_NORMAL, 0.5, (50, 50, 50))

        # ________________________________________________________________ Display the resulting frame
        stack = stack_images(1, [[frame, frame_masked]])
        cv.imshow("Output Stack", stack)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':

    # IP Webcam
    ip = "192.168.1.4"
    port = "8080"

    capture = cv.VideoCapture(f"https://{ip}:{port}/video")

    distance_to_object(5, 28, "smaller")

    capture.release()
    cv.destroyAllWindows()
