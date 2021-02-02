from utils.images_stack import *
from utils.color_filter import *
from utils.thresholds_trackbars import *

grid_img = cv.imread("images/grid.PNG")
corners = []


def map_to_grid(x, y, w, h):
    grid_x = None
    grid_y = None

    if 0 < x < w / 9:
        grid_x = 0
    elif w / 9 < x < 2 * w / 9:
        grid_x = 1
    elif 2 * w / 9 < x < 3 * w / 9:
        grid_x = 2
    elif 3 * w / 9 < x < 4 * w / 9:
        grid_x = 3
    elif 4 * w / 9 < x < 5 * w / 9:
        grid_x = 4
    elif 5 * w / 9 < x < 6 * w / 9:
        grid_x = 5
    elif 6 * w / 9 < x < 7 * w / 9:
        grid_x = 6
    elif 7 * w / 9 < x < 8 * w / 9:
        grid_x = 7
    elif 8 * w / 9 < x < w:
        grid_x = 8

    if 0 < y < h / 3:
        grid_y = 0
    elif h / 3 < y < 2 * h / 3:
        grid_y = 1
    elif 2 * h / 3 < y < h:
        grid_y = 2

    return grid_x, grid_y


def shape_detection(img):

    global grid_img

    h, w = img.shape[:2]
    threshold_trackbars_create()
    while True:

        # Trackbars Values
        blur, b, c, area, epsilon, done = threshold_trackbars_pos()

        # Frame Operations
        frame = cv.resize(img, (0, 0), fx=1, fy=1)
        frame_grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_blur = cv.medianBlur(frame_grey, blur)
        frame_threshold = cv.adaptiveThreshold(frame_blur, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, b, c)

        contours, _ = cv.findContours(frame_threshold, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        for contour_index in range(len(contours)):
            # Check if the contour is relevant
            if area < cv.contourArea(contours[contour_index]) < 5000:
                # Number of Corners
                perimeter = cv.arcLength(contours[contour_index], True)
                poly = cv.approxPolyDP(contours[contour_index], epsilon * perimeter, True)
                bbox_x, bbox_y, bbox_w, bbox_h = cv.boundingRect(poly)
                cv.rectangle(frame, (bbox_x, bbox_y), (bbox_x + bbox_w, bbox_y + bbox_h), (0, 0, 0), 2)

                num_corners = len(poly)
                cv.putText(frame, f"{num_corners}", (int(bbox_x + bbox_w / 2), int(bbox_y + bbox_h / 2)),
                           cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)

                if done == 1:
                    if num_corners == 5:
                        x, y = map_to_grid(int(bbox_x + bbox_w / 2), int(bbox_y + bbox_h / 2), w, h)
                        overlay_grid(x, y, (255, 0, 0))
                    elif num_corners == 4:
                        if 0.9 <= bbox_w / bbox_h <= 1.1:
                            x, y = map_to_grid(int(bbox_x + bbox_w / 2), int(bbox_y + bbox_h / 2), w, h)
                            overlay_grid(x, y, (0, 255, 255))

                        else:
                            x, y = map_to_grid(int(bbox_x + bbox_w / 2), int(bbox_y + bbox_h / 2), w, h)
                            overlay_grid(x, y, (0, 0, 255))

                    else:
                        x, y = map_to_grid(int(bbox_x + bbox_w / 2), int(bbox_y + bbox_h / 2), w, h)
                        overlay_grid(x, y, (0, 255, 0))

                    cv.imshow("output", grid_img)

        stack = stack_images(1, [[frame, frame_grey], [frame_blur, frame_threshold]])
        cv.imshow("frame", stack)
        cv.waitKey(1)


def overlay_grid(x, y, color):
    global grid_img

    if x is None or y is None:
        return

    x_s = [119, 303, 489, 675, 860, 1044, 1230, 1416, 1598]
    y_s = [114, 299, 488]

    cv.circle(grid_img, (x_s[x], y_s[y]), 60, color, 2)


def mouse_click_pos(event, x, y, unused_1, unused_2):
    global corners
    if event == cv.EVENT_LBUTTONDOWN:
        corners.append((x, y))


def distance_between_2_points(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    return int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))


def crop_img(img):

    global corners

    while len(corners) != 3:
        cv.imshow("image", img)
        cv.namedWindow("image")
        cv.setMouseCallback("image", mouse_click_pos)
        cv.waitKey(1)

        print(corners)

    x, y = corners[0]
    w = distance_between_2_points(corners[0], corners[1])
    h = distance_between_2_points(corners[0], corners[2])
    return img[y:y + h, x:x + w]


img = cv.imread("images/bottom.PNG")
new = crop_img(img)
shape_detection(new)