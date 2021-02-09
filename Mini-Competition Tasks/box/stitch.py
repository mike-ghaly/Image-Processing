import cv2
import numpy as np
from imutils import perspective
import imutils


def imgscal(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img



def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def concat_tile_resize(im_list_2d, interpolation=cv2.INTER_CUBIC):
    im_list_v = [hconcat_resize_min(im_list_h, interpolation=cv2.INTER_CUBIC) for im_list_h in im_list_2d]
    return vconcat_resize_min(im_list_v, interpolation=cv2.INTER_CUBIC)

def getPerspectiveTransform(img) :
    pre = img
    gray = cv2.cvtColor(pre.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    canny = cv2.Canny(gray.copy(), 10, 10)
    Canny = cv2.dilate(canny, kernel, iterations=1)
    Canny = cv2.erode(canny, kernel, iterations=1)
    cnts, _ = cv2.findContours(canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    c = max(cnts, key=cv2.contourArea)

    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(pre) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(pre, [box], -1, (0, 255, 0), 2)
    # cv2.drawContours(pre, cnts , -1, (0, 255, 0), 2)

    box = cv2.boxPoints(rect)
    box = np.array(box)
    box = perspective.order_points(box.astype(int))
    (tl, tr, br, bl) = box

    pts1 = np.float32([tl, bl, tr, br])
    x = img.shape[1]
    y = img.shape[0]
    pts2 = np.float32([[0, 0], [0, y], [x,0], [x, y]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(pre, matrix, (x,y))
    return result

def crop(img ,dir):
    if dir=="up" :
        x=0
        y=0
        w=img.shape[1]
        h=20
    elif dir=="down" :
        x = 0
        y = img.shape[0]-20
        w = img.shape[1]
        h = img.shape[0]
    elif dir == "right" :
        y=0
        h=img.shape[0]
        x=img.shape[1]-20
        w = img.shape[1]
    elif dir == "left" :
        y = 0
        h = img.shape[0]
        x= 0
        w=20
    else:
        print("please enter valid direction")
        return
    crop_img = img[y:y + h, x:x + w]
    blanck = np.zeros(img.shape, np.uint8)
    blanck[y:y + h, x:x + w] = crop_img
    blanck[y:y + h, x:x + w] = 255

    return cv2.cvtColor(blanck,cv2.COLOR_BGR2GRAY)


def histCompareImages(src , firstImg ,secondImg , comMask , firstMask , secondMask) :
    # gray_image = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([src], [0],
                             comMask.astype(np.uint8), [256], [0, 256])

    # data1 image
    # gray_image1 = cv2.cvtColor(_1, cv2.COLOR_BGR2GRAY)
    histogram1 = cv2.calcHist([firstImg], [0],
                              firstMask.astype(np.uint8), [256], [0, 256])

    # data2 image
    # gray_image2 = cv2.cvtColor(_2, cv2.COLOR_BGR2GRAY)
    histogram2 = cv2.calcHist([secondImg], [0],
                              secondMask.astype(np.uint8), [256], [0, 256])

    c1, c2 = 0, 0

    # Euclidean Distace between data1 and test
    i = 0
    while i < len(histogram) and i < len(histogram1):
        c1 += (histogram[i] - histogram1[i]) ** 2
        i += 1
    c1 = c1 ** (1 / 2)

    # Euclidean Distace between data2 and test
    i = 0
    while i < len(histogram) and i < len(histogram2):
        c2 += (histogram[i] - histogram2[i]) ** 2
        i += 1
    c2 = c2 ** (1 / 2)

    if (c1 < c2):
        # print("data1.jpg is more similar to test.jpg as compare to data2.jpg")
        # print("c1 = {} , c2 = {}",c1,c2)
        return firstImg , firstMask
    else:
        # print("data2.jpg is more similar to test.jpg as compare to data1.jpg")
        # print("c1 = {} , c2 = {}", c1, c2)
        return secondImg , secondMask


def stitch(topImg, _2, _3, _4, _5):
    # _1 = getPerspectiveTransform(topImg)
    # _2 = getPerspectiveTransform(_2)
    # _3 = getPerspectiveTransform(_3)
    # _4 = getPerspectiveTransform(_4)
    # _5 = getPerspectiveTransform(_5)

    blanck = np.zeros(_1.shape, np.uint8)

    com = _1
    comMask = crop(com, "down")
    _2Mask = crop(_2, "up")
    _3Mask = crop(_3, "up")
    _4Mask = crop(_4, "up")
    _5Mask = crop(_5, "up")
    ret, retMask = histCompareImages(com, _2, _3, comMask, _2Mask, _3Mask)
    ret, retMask = histCompareImages(com, ret, _4, comMask, retMask, _4Mask)
    ret, retMask = histCompareImages(com, ret, _5, comMask, retMask, _5Mask)

    final_3 = ret

    com = final_3
    comMask = crop(final_3, "left")
    _2Mask = crop(_2, "right")
    _3Mask = crop(_3, "right")
    _4Mask = crop(_4, "right")
    _5Mask = crop(_5, "right")
    ret, retMask = histCompareImages(com, _2, _3, comMask, _2Mask, _3Mask)
    ret, retMask = histCompareImages(com, ret, _4, comMask, retMask, _4Mask)
    ret, retMask = histCompareImages(com, ret, _5, comMask, retMask, _5Mask)

    final_2 = ret

    com = final_3
    comMask = crop(final_3, "right")
    _2Mask = crop(_2, "left")
    _3Mask = crop(_3, "left")
    _4Mask = crop(_4, "left")
    _5Mask = crop(_5, "left")
    ret, retMask = histCompareImages(com, _2, _3, comMask, _2Mask, _3Mask)
    ret, retMask = histCompareImages(com, ret, _4, comMask, retMask, _4Mask)
    ret, retMask = histCompareImages(com, ret, _5, comMask, retMask, _5Mask)

    final_4 = ret

    com = final_4
    comMask = crop(final_4, "right")
    _2Mask = crop(_2, "left")
    _3Mask = crop(_3, "left")
    _4Mask = crop(_4, "left")
    _5Mask = crop(_5, "left")
    ret, retMask = histCompareImages(com, _2, _3, comMask, _2Mask, _3Mask)
    ret, retMask = histCompareImages(com, ret, _4, comMask, retMask, _4Mask)
    ret, retMask = histCompareImages(com, ret, _5, comMask, retMask, _5Mask)

    final_5 = ret

    _2 = final_2
    _3 = final_3
    _4 = final_4
    _5 = final_5

    im_tile_resize = concat_tile_resize([[blanck, _1, blanck, blanck],
                                         [_2, _3, _4,_5]])
    im_tile_resize = imgscal(im_tile_resize, 50)
    cv2.imshow("After the Model", im_tile_resize)
    cv2.waitKey(0)

# images from screenshots
# _1 = cv2.imread("photomosaic/1photomosaic.png")
# _4 = cv2.imread("photomosaic/2photomosaic.png")
# _5 = cv2.imread("photomosaic/3photomosaic.png")
# _3 = cv2.imread("photomosaic/4photomosaic.png")
# _2 = cv2.imread("photomosaic/5photomosaic.png")

_1 = cv2.imread("photomosaic/1.jpeg")
_4 = cv2.imread("photomosaic/2.jpeg")
_5 = cv2.imread("photomosaic/3.jpeg")
_3 = cv2.imread("photomosaic/4.jpeg")
_2 = cv2.imread("photomosaic/5.jpeg")



stitch(_1,_2,_3,_4,_5)
