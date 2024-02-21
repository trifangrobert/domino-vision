import pickle
import cv2 as cv
import numpy as np
from pprint import pprint
import os
from tqdm import tqdm

class BoardExtractor:
    def __init__(self, hparams_path: str, width: int = 900, height: int = 900, expand_size: int = 10) -> None:
        with open(hparams_path, "rb") as f:
            self.hparams = pickle.load(f)
        self.width = width
        self.height = height
        self.expand_size = expand_size

    def view_config(self):
        pprint(self.hparams)

    def extract_board(self, image: np.ndarray):
        # print(image.shape)
        # image = cv.resize(image, (0, 0), fx=0.2, fy=0.2)

        original_image = image.copy()
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        guassian_blur_kernel = self.hparams["guassian_blur_kernel"]
        median_blur_kernel = self.hparams["median_blur_kernel"]
        threshold = self.hparams["threshold"]
        erode_kernel_size = self.hparams["erode_kernel_size"]
        erode_iterations = self.hparams["erode_iterations"]
        dilate_kernel_size = self.hparams["dilate_kernel_size"]
        dilate_iterations = self.hparams["dilate_iterations"]
        canny_min = self.hparams["canny_min"]
        canny_max = self.hparams["canny_max"]
        hue_min = self.hparams["hue_min"]
        hue_max = self.hparams["hue_max"]
        saturation_min = self.hparams["saturation_min"]
        saturation_max = self.hparams["saturation_max"]
        value_min = self.hparams["value_min"]
        value_max = self.hparams["value_max"]


        image = cv.inRange(image, (hue_min, saturation_min, value_min), (hue_max, saturation_max, value_max))
        image = cv.GaussianBlur(image, (2 * guassian_blur_kernel + 1, 2 * guassian_blur_kernel + 1), 0)
        image = cv.medianBlur(image, 2 * median_blur_kernel + 1)
        _, image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
        kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        image = cv.erode(image, kernel, iterations=erode_iterations)
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        image = cv.dilate(image, kernel, iterations=dilate_iterations)
        image = cv.Canny(image, canny_min, canny_max)
        contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) 

        max_area = 0
        top_left = None
        bottom_right = None
        top_right = None
        bottom_left = None 

        for i in range(len(contours)):
            if(len(contours[i]) > 3):
                possible_top_left = None
                possible_bottom_right = None
                for point in contours[i].squeeze():
                    if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                        possible_top_left = point

                    if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                        possible_bottom_right = point

                diff = np.diff(contours[i].squeeze(), axis = 1)
                possible_top_right = contours[i].squeeze()[np.argmin(diff)]
                possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
                if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                    max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                    top_left = possible_top_left
                    bottom_right = possible_bottom_right
                    top_right = possible_top_right
                    bottom_left = possible_bottom_left


        if max_area > 0:
            # board game
            puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
            destination_of_puzzle = np.array([[0, 0],[self.width, 0],[self.width, self.height],[0, self.height]], dtype = "float32")
            M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)
            result = cv.warpPerspective(original_image, M, (self.width, self.height))

            
            # expanded board game
            expand_top_left = (top_left[0] - self.expand_size, top_left[1] - self.expand_size)
            expand_top_right = (top_right[0] + self.expand_size, top_right[1] - self.expand_size)
            expand_bottom_right = (bottom_right[0] + self.expand_size, bottom_right[1] + self.expand_size)
            expand_bottom_left = (bottom_left[0] - self.expand_size, bottom_left[1] + self.expand_size)

            puzzle = np.array([expand_top_left,expand_top_right,expand_bottom_right,expand_bottom_left], dtype = "float32")
            destination_of_puzzle = np.array([[0, 0],[self.width + 2 * self.expand_size, 0],[self.width + 2 * self.expand_size, self.height + 2 * self.expand_size],[0, self.height + 2 * self.expand_size]], dtype = "float32")
            M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)
            result_expanded = cv.warpPerspective(original_image, M, (self.width + 2 * self.expand_size, self.height + 2 * self.expand_size))
            # cv.imshow("Board", result)
            # cv.imshow("Board expanded", result_expanded)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            return result, result_expanded

            # x_min = min(top_left[0], top_right[0])
            # x_max = max(bottom_left[0], bottom_right[0])
            # y_min = min(top_left[1], bottom_left[1])
            # y_max = max(top_right[1], bottom_right[1])

            # top_left = (x_min, y_min)
            # top_right = (x_min, y_max)
            # bottom_left = (x_max, y_min)
            # bottom_right = (x_max, y_max)

            # corners_image = original_image.copy()
            # cv.circle(corners_image, top_left, 5, (0, 0, 255), -1)
            # cv.circle(corners_image, top_right, 5, (0, 0, 255), -1)
            # cv.circle(corners_image, bottom_left, 5, (0, 0, 255), -1)
            # cv.circle(corners_image, bottom_right, 5, (0, 0, 255), -1)

            # # cv.imshow("Corners", corners_image)

            # result = original_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
            # result = cv.resize(result, (self.width, self.height))

            # # cv.imshow("Board", result)

            # result_expanded = original_image[y_min - self.expand_size:y_max + self.expand_size, x_min - self.expand_size:x_max + self.expand_size]
            # print(result.shape)
            # print(result_expanded.shape)
            # result_expanded = cv.resize(result_expanded, (self.width + 2 * self.expand_size, self.height + 2 * self.expand_size))

            # # cv.imshow("Board expanded", result_expanded)
            # # cv.waitKey(0)
            # # cv.destroyAllWindows()

            # return result, result_expanded

        return None, None
    
if __name__ == "__main__":
    # image = cv.imread("../fake_test/1_08.jpg")

    board_extractor = BoardExtractor("config9.pickle", 900, 900, 10)
    # board_extractor = BoardExtractor("domino_config1.pickle", 900, 900)
    # board_extractor.view_config()

    game_path = "../../antrenare/"
    # game_path = "../fake_test/"
    game_index = 1
    game_files = sorted([file for file in os.listdir(game_path) if file.startswith(f"{game_index}_") and file.endswith(".jpg")])
    for file in game_files:
        if file.endswith(".jpg"):
            image = cv.imread(os.path.join(game_path, file))
            board, expanded_board = board_extractor.extract_board(image)
            cv.imshow("Board", board)
            cv.imshow("Expanded board", expanded_board)
            cv.waitKey(0)
    cv.destroyAllWindows()

    # game_path = "../../antrenare/"
    # for file in tqdm(os.listdir(game_path)):
    #     if file.endswith(".jpg"):
    #         image = cv.imread(os.path.join(game_path, file))
    #         board, expanded_board = board_extractor.extract_board(image)
    #         cv.imwrite(f"../../antrenare_board_hq/{file}", board)
    
