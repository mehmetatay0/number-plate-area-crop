import cv2 as cv
import numpy as np
import dfunc as df


def main():
    # image reading with cars list
    car_list = np.array(['cars/car1.jpg', 'cars/car2.jpg', 'cars/car3.jpg'])
    for car in car_list:
        img = cv.imread(car)

        # Finding Number Plate
        edged = df.EdgeDetection(img)
        contour = df.FindingContour(edged)
        if contour.any():
            sortedPts = df.SortingPoints(contour)
            cropped_img = df.Transform(img, sortedPts)
            cv.imwrite('numberplates/plate_' + car, cropped_img)
    return None


if __name__ == "__main__":
    main()