from utils.images_stack import *
from scipy.spatial import distance as dist


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


def corners(polygon):
    return np.float32([polygon[0][0], polygon[3][0], polygon[1][0], polygon[2][0]])


font = cv.FONT_HERSHEY_COMPLEX

img = cv.imread("Untitled.png")
img = cv.resize(img, (0, 0), fx=0.5, fy=0.5)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 2)
img_output = img.copy()

contours, _ = cv.findContours(img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
relevant_contours_list = []
for contour_index in range(len(contours)):
    if 800 < cv.contourArea(contours[contour_index]) < 20000:
        relevant_contours_list.append(contours[contour_index])

output_imgs = []
print(len(relevant_contours_list))
for contour_index in range(len(relevant_contours_list)):

    box = cv.minAreaRect(relevant_contours_list[contour_index])
    box = np.array(box, dtype="int")

    old_corners = order_points(box)

    w = abs(old_corners[0][0] - old_corners[1][0])
    h = abs(old_corners[0][1] - old_corners[3][1])

    new_corners = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    transformation_matrix = cv.getPerspectiveTransform(old_corners, new_corners)
    output_imgs.append(cv.warpPerspective(img, transformation_matrix, (w, h)))

cv.imshow("DA", img_output)
cv.waitKey(0)
