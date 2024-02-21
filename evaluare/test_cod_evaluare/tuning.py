import numpy as np
import cv2 as cv
import argparse
import pickle

from utils import draw_grid, draw_corners, get_corners

def create_trackbar(frame: np.ndarray, hparams: dict = None):
    width = 900
    height = 900
    expand_size = 10

    # frame = cv.resize(frame, (0, 0), fx=0.7, fy=0.7) # for pieces

    # frame = cv.resize(frame, (0, 0), fx=0.2, fy=0.2) # for board
    # frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5) # for board

    original_frame = frame.copy()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    def nothing(x):
        pass

    if hparams is None:
        print("Using default parameters")
        hparams = {}
        guassian_blur_kernel = 0
        median_blur_kernel = 0
        threshold = 0
        erode_kernel_size = 0
        erode_iterations = 0
        dilate_kernel_size = 0
        dilate_iterations = 0
        canny_min = 0
        canny_max = 255
        hue_min = 0
        hue_max = 255
        saturation_min = 0
        saturation_max = 255
        value_min = 0
        value_max = 255
    else:
        print("Using loaded parameters")
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

    cv.namedWindow('Trackbar')
        
    cv.createTrackbar('GaussianBlur', 'Trackbar', guassian_blur_kernel, 20, nothing)
    cv.createTrackbar('MedianBlur', 'Trackbar', median_blur_kernel, 20, nothing)
    cv.createTrackbar('Threshold', 'Trackbar', threshold, 255, nothing)
    cv.createTrackbar('Erode Kernel Size', 'Trackbar', erode_kernel_size, 10, nothing)
    cv.createTrackbar('Erode Iterations', 'Trackbar', erode_iterations, 20, nothing)
    cv.createTrackbar('Dilate Kernel Size', 'Trackbar', dilate_kernel_size, 20, nothing)
    cv.createTrackbar('Dilate Iterations', 'Trackbar', dilate_iterations, 20, nothing)
    cv.createTrackbar('Canny Min', 'Trackbar', canny_min, 255, nothing)
    cv.createTrackbar('Canny Max', 'Trackbar', canny_max, 255, nothing)
    cv.createTrackbar('Hue Min', 'Trackbar', hue_min, 255, nothing)
    cv.createTrackbar('Hue Max', 'Trackbar', hue_max, 255, nothing)
    cv.createTrackbar('Saturation Min', 'Trackbar', saturation_min, 255, nothing)
    cv.createTrackbar('Saturation Max', 'Trackbar', saturation_max, 255, nothing)
    cv.createTrackbar('Value Min', 'Trackbar', value_min, 255, nothing)
    cv.createTrackbar('Value Max', 'Trackbar', value_max, 255, nothing)


    
    while True:
        guassian_blur_kernel = cv.getTrackbarPos('GaussianBlur', 'Trackbar')
        median_blur_kernel = cv.getTrackbarPos('MedianBlur', 'Trackbar')
        threshold = cv.getTrackbarPos('Threshold', 'Trackbar')
        erode_kernel_size = cv.getTrackbarPos('Erode Kernel Size', 'Trackbar')
        erode_iterations = cv.getTrackbarPos('Erode Iterations', 'Trackbar')
        dilate_kernel_size = cv.getTrackbarPos('Dilate Kernel Size', 'Trackbar')
        dilate_iterations = cv.getTrackbarPos('Dilate Iterations', 'Trackbar')
        canny_min = cv.getTrackbarPos('Canny Min', 'Trackbar')
        canny_max = cv.getTrackbarPos('Canny Max', 'Trackbar')
        hue_min = cv.getTrackbarPos('Hue Min', 'Trackbar')
        hue_max = cv.getTrackbarPos('Hue Max', 'Trackbar')
        saturation_min = cv.getTrackbarPos('Saturation Min', 'Trackbar')
        saturation_max = cv.getTrackbarPos('Saturation Max', 'Trackbar')
        value_min = cv.getTrackbarPos('Value Min', 'Trackbar')
        value_max = cv.getTrackbarPos('Value Max', 'Trackbar')

        hparams["guassian_blur_kernel"] = guassian_blur_kernel
        hparams["median_blur_kernel"] = median_blur_kernel
        hparams["threshold"] = threshold
        hparams["erode_kernel_size"] = erode_kernel_size
        hparams["erode_iterations"] = erode_iterations
        hparams["dilate_kernel_size"] = dilate_kernel_size
        hparams["dilate_iterations"] = dilate_iterations
        hparams["canny_min"] = canny_min
        hparams["canny_max"] = canny_max
        hparams["hue_min"] = hue_min
        hparams["hue_max"] = hue_max
        hparams["saturation_min"] = saturation_min
        hparams["saturation_max"] = saturation_max
        hparams["value_min"] = value_min
        hparams["value_max"] = value_max

        # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # gray_blurred = cv.GaussianBlur(gray, (9, 9), 2, 2)
    
        mask = cv.inRange(frame, (hue_min, saturation_min, value_min), (hue_max, saturation_max, value_max))
        blur = cv.GaussianBlur(mask, (2 * guassian_blur_kernel + 1, 2 * guassian_blur_kernel + 1), 0)
        blur = cv.medianBlur(blur, 2 * median_blur_kernel + 1)
        _, thresh = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)
        kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        thresh = cv.erode(thresh, kernel, iterations=erode_iterations)
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        thresh = cv.dilate(thresh, kernel, iterations=dilate_iterations)
        edges = cv.Canny(thresh, canny_min, canny_max)
        contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # detected_circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, 20, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)

        res = cv.bitwise_and(frame, frame, mask=mask)

        # image_circles = original_frame.copy()
        # if detected_circles is not None:
        #     print("Circles detected")
        #     detected_circles = np.uint16(np.around(detected_circles))
        #     for pt in detected_circles[0, :]:
        #         a, b, r = pt[0], pt[1], pt[2]
        #         cv.circle(image_circles, (a, b), r, (0, 255, 0), 2)
        #         cv.circle(image_circles, (a, b), 1, (0, 0, 255), 3)
        # else:
        #     print("No circles detected")

        print_frame = cv.cvtColor(frame, cv.COLOR_HSV2BGR)
        print_mask = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)
        print_blur = cv.cvtColor(blur, cv.COLOR_GRAY2BGR)
        print_edges = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        print_res = cv.cvtColor(res, cv.COLOR_HSV2BGR)
        print_contours = cv.drawContours(original_frame.copy(), contours, -1, (0, 255, 0), 3)
        print_thresh = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        
        is_domino = True
        if not is_domino:
            print_frame = cv.resize(print_frame, (width, height))
            print_mask = cv.resize(print_mask, (width, height))
            print_blur = cv.resize(print_blur, (width, height))
            print_edges = cv.resize(print_edges, (width, height))
            print_res = cv.resize(print_res, (width, height))
            print_contours = cv.resize(print_contours, (width, height))
            print_thresh = cv.resize(print_thresh, (width, height))

        # cv.imshow('Original', print_frame)
        cv.imshow('Mask', print_mask)
        # cv.imshow('Blur', print_blur)
        # cv.imshow('Edges', print_edges)
        # cv.imshow('Result', print_res)
        cv.imshow('Contours', print_contours)
        cv.imshow('Threshold', print_thresh)
        # cv.imshow('Circles', image_circles)

        top_left, top_right, bottom_right, bottom_left, max_area = get_corners(contours)

        max_area = 0 # for dominoes

        if max_area > 0:
            image_corners = draw_corners(frame, [top_left, top_right, bottom_right, bottom_left])
            # cv.imshow("Image with corners", image_corners)

            
            expand_top_left = (top_left[0] - expand_size, top_left[1] - expand_size)
            expand_top_right = (top_right[0] + expand_size, top_right[1] - expand_size)
            expand_bottom_right = (bottom_right[0] + expand_size, bottom_right[1] + expand_size)
            expand_bottom_left = (bottom_left[0] - expand_size, bottom_left[1] + expand_size)

            puzzle = np.array([expand_top_left,expand_top_right,expand_bottom_right,expand_bottom_left], dtype = "float32")
            destination_of_puzzle = np.array([[0, 0],[width + expand_size, 0],[width + expand_size, height + expand_size],[0, height + expand_size]], dtype = "float32")
            M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)
            result = cv.warpPerspective(frame, M, (width + expand_size, height + expand_size))
            result = cv.cvtColor(result, cv.COLOR_HSV2BGR)

            result = cv.resize(result, (width, height))
            cv.imshow("Result", result)
            
            # puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
            # destination_of_puzzle = np.array([[0, 0],[width, 0],[width, height],[0, height]], dtype = "float32")
            # M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)
            # result = cv.warpPerspective(frame, M, (width, height))
            # result = cv.cvtColor(result, cv.COLOR_HSV2BGR)
            # image_grid = cv.cvtColor(result, cv.COLOR_BGR2HSV)

            # image_grid = draw_grid(result)
            # cv.imshow("Result", result)
            # cv.imshow("Image with grid", image_grid)


        if cv.waitKey(25) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
    return hparams


def load_parameters(file_path: str):

    with open(file_path, "rb") as f:
        hparams = pickle.load(f)
    return hparams

def save_parameters(file_path: str, hparams: dict):
    with open(file_path, "wb") as f:
        pickle.dump(hparams, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning parameters')
    parser.add_argument('-l', '--load', type=str, help='Load config file')
    parser.add_argument('-s', '--save', type=str, help='Save config file')
    parser.add_argument('-i', '--image', type=str, help='Image to tune on')
    args = parser.parse_args()

    if args.image:
        image = cv.imread(args.image)
        print("Tuning on image: ", args.image)
    else:
        print("No image specified, crashing to avoid overwriting config file")
        exit(1)

    # image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    hparams = None
    if args.load:
        hparams = load_parameters(args.load)
        print(f"Loaded parameters from {args.load}")
    
    hparams = create_trackbar(image, hparams)

    print(hparams)

    if args.save:
        save_parameters(args.save, hparams)
        print(f"Saved parameters to {args.save}")


