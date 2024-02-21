import numpy as np
import cv2 as cv
import argparse
import pickle

from utils import get_corners

def draw_corners(image: np.ndarray, hparams: dict):
    original_image = image.copy()

    width = 900
    height = 900


    guassian_blur_kernel = hparams["guassian_blur_kernel"]
    median_blur_kernel = hparams["median_blur_kernel"]
    threshold = hparams["threshold"]
    erode_kernel_size = hparams["erode_kernel_size"]
    erode_iterations = hparams["erode_iterations"]
    dilate_kernel_size = hparams["dilate_kernel_size"]
    dilate_iterations = hparams["dilate_iterations"]
    canny_min = hparams["canny_min"]
    canny_max = hparams["canny_max"]
    hue_min = hparams["hue_min"]
    hue_max = hparams["hue_max"]
    saturation_min = hparams["saturation_min"]
    saturation_max = hparams["saturation_max"]
    value_min = hparams["value_min"]
    value_max = hparams["value_max"]


    image = cv.inRange(image, (hue_min, saturation_min, value_min), (hue_max, saturation_max, value_max))
    image = cv.GaussianBlur(image, (2 * guassian_blur_kernel + 1, 2 * guassian_blur_kernel + 1), 0)
    image = cv.medianBlur(image, 2 * median_blur_kernel + 1)
    _, image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
    image = cv.erode(image, kernel, iterations=erode_iterations)
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    image = cv.dilate(image, kernel, iterations=dilate_iterations)
    image = cv.Canny(image, canny_min, canny_max)
    contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 

    top_left, top_right, bottom_right, bottom_left, max_area = get_corners(contours)


    if max_area > 0:
        # x_min = min(top_left[0], top_right[0])
        # x_max = max(bottom_left[0], bottom_right[0])
        # y_min = min(top_left[1], bottom_left[1])
        # y_max = max(top_right[1], bottom_right[1])

        # top_left = (x_min, y_min)
        # top_right = (x_max, y_min)
        # bottom_right = (x_max, y_max)
        # bottom_left = (x_min, y_max)


        image_copy = cv.cvtColor(original_image.copy(), cv.COLOR_HSV2BGR)
        cv.circle(image_copy,tuple(top_left),5,(0,0,255),-1)
        cv.circle(image_copy,tuple(top_right),5,(0,0,255),-1)
        cv.circle(image_copy,tuple(bottom_left),5,(0,0,255),-1)
        cv.circle(image_copy,tuple(bottom_right),5,(0,0,255),-1)
        image_copy = cv.resize(image_copy, (0, 0), fx=0.2, fy=0.2)
        cv.imshow("Image with corners", image_copy)    

        puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
        destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")
        M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)
        result = cv.warpPerspective(original_image, M, (width, height))
        result = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        

        horizontal_lines = []
        step_width = width / 15
        for i in range(0, 16):
            horizontal_lines.append([(0, int(i * step_width)), (width - 1, int(i * step_width))])

        vertical_lines = []
        step_height = height / 15
        for i in range(0, 16):
            vertical_lines.append([(int(i * step_height), 0), (int(i * step_height), height - 1)])

        for line in horizontal_lines:
            cv.line(result, line[0], line[1], (0, 0, 255), 2)

        for line in vertical_lines:
            cv.line(result, line[0], line[1], (0, 0, 255), 2)

        cv.imshow("Result", result)

        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--hparams", type=str, required=True, help="Path to the hparams file")
    args = parser.parse_args()

    image = cv.imread(args.image)
    image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    # image = cv.resize(image, (0, 0), fx=0.2, fy=0.2)

    with open(args.hparams, "rb") as f:
        hparams = pickle.load(f)

    draw_corners(image, hparams)

    
