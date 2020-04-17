import cv2 as cv
import numpy as np


def EdgeDetection(img):
    # Gray -> Blur -> and Canny Process
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    blur = cv.GaussianBlur(gray, (1, 1), 0)
    # blur = cv.blur(img, (3, 3))

    edged = cv.Canny(blur, 100, 100)
    # edged = cv.Canny(blur, 100, 200)
    return edged


def FindingContour(img):
    global contour
    contour = np.array([])
    contours, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Contour Approximation for Finding Rectangle Objects
        epsilon = 0.01 * cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)

        # this process finding to bigger rectangle area
        area = cv.contourArea(approx)
        if len(approx) == 4 and (cv.contourArea(approx) > cv.arcLength(approx, True)):
            contour = approx
    return contour


def SortingPoints(contour):
    # Points should be resize
    pts = contour.reshape(4, 2)
    sorted = np.zeros((4, 2), dtype="int")

    # top left - bottom right
    sum = np.sum(pts, axis=-1)
    sorted[0] = pts[sum.argmin()]
    sorted[2] = pts[sum.argmax()]

    # top right - bottom left
    diff = np.diff(pts, axis=-1)
    sorted[3] = pts[diff.argmax()]
    sorted[1] = pts[diff.argmin()]

    return sorted

def Transform(img, pts):
    # Finding Width and Height

    widthA = np.sqrt( (pts[1, 0] - pts[0, 0])**2 + (pts[1, 1] - pts[0, 1])**2)
    widthB = np.sqrt( (pts[2, 0] - pts[3, 0])**2 + (pts[2, 1] - pts[3, 1])**2)
    width = int(max(widthA, widthB))

    heightA = np.sqrt((pts[0, 0] - pts[3, 0]) ** 2 + (pts[0, 1] - pts[3, 1]) ** 2)
    heightB = np.sqrt((pts[1, 0] - pts[2, 0]) ** 2 + (pts[1, 1] - pts[2, 1]) ** 2)
    height = int(max(heightA, heightB))

    # Points of be cropped image should calculate
    # Numpy data type should be float !

    dst = np.array([
        [0, 0],
        [width-1, 0],
        [width-1, height-1],
        [0, height-1]
    ], dtype="float32")

    pts = pts.astype("float32")

    PT = cv.getPerspectiveTransform(pts, dst)
    cropped = cv.warpPerspective(img, PT, (width, height))
    return cropped
