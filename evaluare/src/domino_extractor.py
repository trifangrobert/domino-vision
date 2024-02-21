import numpy as np
import cv2 as cv
import pickle
from typing import List

class DominoExtractor:
    def __init__(self, hparams_path: str, width: int = 900, height: int = 900, offset: int = 5, expand_size: int = 5) -> None:
        with open(hparams_path, "rb") as f:
            self.hparams = pickle.load(f)
        self.width = width
        self.height = height
        self.expand_size = expand_size
        self.offset = offset

    def extract_dominoes(self, image: np.ndarray) -> np.ndarray:
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


        mask = cv.inRange(image, (hue_min, saturation_min, value_min), (hue_max, saturation_max, value_max))
        blur = cv.GaussianBlur(mask, (2 * guassian_blur_kernel + 1, 2 * guassian_blur_kernel + 1), 0)
        blur = cv.medianBlur(blur, 2 * median_blur_kernel + 1)
        _, thresh = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)
        kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        thresh = cv.erode(thresh, kernel, iterations=erode_iterations)
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        thresh = cv.dilate(thresh, kernel, iterations=dilate_iterations)

        step_size = self.width // 15

        threshold_percentage = 0.45 # might need to be changed
        threshold_value = threshold_percentage * step_size * step_size

        result = original_image.copy()

        marked = np.zeros((15, 15), dtype=np.uint8)

        for i in range(0, 15):
            for j in range(0, 15):
                x = i * step_size
                y = j * step_size
                roi = thresh[y:y+step_size, x:x+step_size]
                
                cnt_white_pixels = np.sum(roi == 255)
                # print(cnt_white_pixels)
                if cnt_white_pixels > threshold_value:
                    marked[i, j] = 1
                    # print("Domino")
                else:
                    result[y:y+step_size, x:x+step_size] = 0

        # cv.imshow("Original", original_image)
        # cv.imshow("Result", result)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        return marked

    def extract_domino(self, image: np.ndarray, domino_position: List[str]) -> np.ndarray:
        top_left = domino_position[0]
        bottom_right = domino_position[1]

        # top_left and bottom_right will look like this: 8H 8I

        row_top_left = int(top_left[:-1]) - 1
        col_top_left = top_left[-1]

        row_bottom_right = int(bottom_right[:-1]) - 1
        col_bottom_right = bottom_right[-1]

        # print(row_top_left, col_top_left, row_bottom_right, col_bottom_right)
        # print(self.offset)
        step_size = self.width // 15

        row_top_left = self.offset + row_top_left * step_size
        col_top_left = self.offset + (ord(col_top_left) - 65) * step_size

        row_bottom_right = self.offset + (row_bottom_right + 1) * step_size
        col_bottom_right = self.offset + ((ord(col_bottom_right) - 65) + 1) * step_size

        # print(row_top_left, col_top_left, row_bottom_right, col_bottom_right)

        row_top_left = max(0, row_top_left - self.expand_size)
        col_top_left = max(0, col_top_left - self.expand_size)

        row_bottom_right = min(image.shape[0], row_bottom_right + self.expand_size)
        col_bottom_right = min(image.shape[1], col_bottom_right + self.expand_size)

        # print(row_top_left, col_top_left, row_bottom_right, col_bottom_right)

        domino = image[row_top_left:row_bottom_right, col_top_left:col_bottom_right]

        # cv.imshow("Original", image)
        # cv.imshow("Domino", domino)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        cv.imwrite("../../.jpg", domino)
        
        return domino

if __name__ == "__main__":
    extractor = DominoExtractor("extract_domino_config.pickle", 900, 900, 5, 3)
    image = cv.imread("../../antrenare_board_hq/1_20.jpg")
    extractor.extract_dominoes(image)




        