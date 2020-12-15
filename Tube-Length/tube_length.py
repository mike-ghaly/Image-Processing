import cv2 as cv
import numpy as np


def stack_images(resize_scale, img_array):

    """
        This function is used to display multiple images as a grid
        on the same window.

        Inputs:

            - resize_scale:         (float) (0 < resize_scale <= 1)
                                    resizes all images according to the given scale.

            - img_array:            (2D Array) Images to stack.
                                    ( [[00, 01, 02],
                                       [10, 11, 12],
                                       [20, 21, 22]] )

        Returns:
            Stacked Image Ready for the imshow() function.
    """

    num_rows = len(img_array)
    num_cols = len(img_array[0])

    """ Resize images in array (Grayscale images must also be converted to BGR) """
    processed_array = []
    for row in range(num_rows):

        horizontal_stack = []
        for col in range(num_cols):

            # Resize according to the given scale
            resized_img = cv.resize(img_array[row][col], (0, 0), fx=resize_scale, fy=resize_scale)

            # Color-Space Conversion
            if len(resized_img.shape) == 3:
                horizontal_stack.append(resized_img)
            else:
                horizontal_stack.append(cv.cvtColor(resized_img, cv.COLOR_GRAY2BGR))
        processed_array.append(horizontal_stack)

    """ Stacks the processed images """
    blank_img = np.zeros(processed_array[0][0].shape, np.uint8)
    horizontal_blank_stack = [blank_img] * num_rows
    for row in range(num_rows):
        horizontal_blank_stack[row] = np.hstack(processed_array[row])
    return np.vstack(horizontal_blank_stack)


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


def red_tube_length(image_path, blue_tube_length):

    """ Returns the length of a tube given it's width and height regardless of orientation """

    # Read Image
    img_original = cv.imread(image_path)
    img_original = cv.resize(img_original, (0, 0), fx=0.4, fy=0.4)
    img_blue = img_original.copy()
    img_red = img_original.copy()
    img_hsv = cv.cvtColor(img_original, cv.COLOR_BGR2HSV)

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

    """ Blue Tube """
    lower_bound = np.array([100, 50, 0])
    upper_bound = np.array([170, 255, 255])
    blue_mask = cv.inRange(img_hsv, lowerb=lower_bound, upperb=upper_bound)
    blue_tube_size = None

    # Use Morph_Open to erode and then dilate the image so noise is removed
    kernel = np.ones((9, 9), np.uint8)
    blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_OPEN, kernel)
    # Use Morph_Close to dilate and then erode the image so gaps in tubes are filled
    kernel = np.ones((25, 25), np.uint8)
    blue_mask = cv.morphologyEx(blue_mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(blue_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour_index in range(len(contours)):

        # Check if the contour is relevant
        if cv.arcLength(contours[contour_index], True) > 100:
            # Find the bounding rectangle with minimum area
            rect = cv.minAreaRect(contours[contour_index])
            blue_tube_size = rect[1]
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(img_blue, [box], 0, (255, 0, 150), 2)

    """ Red Tube """
    lower_bound = np.array([0, 90, 0])
    upper_bound = np.array([5, 255, 255])
    red_mask = cv.inRange(img_hsv, lowerb=lower_bound, upperb=upper_bound)
    red_tube_size = None

    # Use Morph_Open to erode and then dilate the image so noise is removed
    kernel = np.ones((9, 9), np.uint8)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_OPEN, kernel)
    # Use Morph_Close to dilate and then erode the image so gaps in tubes are filled
    kernel = np.ones((50, 50), np.uint8)
    red_mask = cv.morphologyEx(red_mask, cv.MORPH_CLOSE, kernel)

    contours, _ = cv.findContours(red_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour_index in range(len(contours)):

        # Check if the contour is relevant
        if cv.arcLength(contours[contour_index], True) > 100:
            # Find the bounding rectangle with minimum area
            rect = cv.minAreaRect(contours[contour_index])
            red_tube_size = rect[1]
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(img_red, [box], 0, (150, 0, 255), 2)

    """ Tube Length Calculation """
    cm_per_pixel = blue_tube_length / max(*blue_tube_size)
    red_tube_length = round(cm_per_pixel * max(*red_tube_size), 3)

    cv.putText(img_blue, f"{blue_tube_length}cm", (20, 40), cv.QT_FONT_NORMAL, 1, (255, 0, 0))
    cv.putText(img_red, f"{red_tube_length}cm", (20, 40), cv.QT_FONT_NORMAL, 1, (0, 0, 255))

    cv.imshow("Tube Length Stack", stack_images(0.4, [[blue_mask, red_mask], [img_blue, img_red]]))
    cv.waitKey(0)


if __name__ == '__main__':
    test_image_path = "len_7.2.jpg"
    blue_tube_length = 7.2
    red_tube_length(test_image_path, blue_tube_length)
