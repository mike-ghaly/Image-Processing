from utils.images_stack import *
from scipy.spatial import distance as dist


def warp(img, object_corners, fx, fy):

    img_width = img.shape[0]
    img_height = img.shape[1]

    img_corners = [
        [0, 0],
        [img_width, 0],
        [0, img_height],
        [img_width, img_height]
    ]

    img_corners = np.float32(img_corners)
    object_corners = np.float32(object_corners)

    transform_matrix = cv.getPerspectiveTransform(object_corners, img_corners)
    object_img = cv.warpPerspective(img, transform_matrix, (img_width, img_height))
    object_img = cv.resize(object_img, (0, 0), fx=fx, fy=fy)

    return object_img


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, bl, br], dtype="float32")


img = cv.imread("box.PNG")
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_binary = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 80)

contours, _ = cv.findContours(img_binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
relevant_contours = []
for contour_index in range(len(contours)):
    if 1000 < cv.contourArea(contours[contour_index]) < 20000:
        relevant_contours.append(contours[contour_index])

output_imgs = []
for contour_index in range(len(relevant_contours)):
    epsilon = 0.02 * cv.arcLength(relevant_contours[contour_index], True)
    poly = cv.approxPolyDP(relevant_contours[contour_index], epsilon, True)

    poly_corners = np.array(poly.reshape(4, 2), dtype="int")
    poly_corners_ordered = order_points(poly_corners)

    if contour_index == 0:
        output_imgs.append(warp(img, poly_corners_ordered, 6, 1))
    elif contour_index == 1:
        output_imgs.append(warp(img, poly_corners_ordered, 2, 1))
    elif contour_index == 2:
        poly_corners_ordered[1], poly_corners_ordered[3] = poly_corners_ordered[3], poly_corners_ordered[1].copy()
        output_imgs.append(warp(img, poly_corners_ordered, 6, 1))

for i in output_imgs:
    cv.imshow("Output Stack", i)
    cv.waitKey(0)
