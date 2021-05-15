
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

