import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import pickle

class DominoCleaner:
    def __init__(self, hparams_path: str, width: int = 60, height: int = 120) -> None:
        with open(hparams_path, "rb") as f:
            self.hparams = pickle.load(f)
        self.width = width
        self.height = height

        # print("Loaded parameters: ", self.hparams)


    def clean_domino(self, domino: np.ndarray) -> np.ndarray:
        # domino = cv.resize(domino, (self.width, self.height))
        domino = cv.cvtColor(domino, cv.COLOR_BGR2HSV)
        # domino = cv.resize(domino, (self.width, self.height))

        guassian_blur_kernel = self.hparams["guassian_blur_kernel"]
        median_blur_kernel = self.hparams["median_blur_kernel"]
        threshold = self.hparams["threshold"]
        erode_kernel_size = self.hparams["erode_kernel_size"]
        erode_iterations = self.hparams["erode_iterations"]
        dilate_kernel_size = self.hparams["dilate_kernel_size"]
        dilate_iterations = self.hparams["dilate_iterations"]
        hue_min = self.hparams["hue_min"]
        hue_max = self.hparams["hue_max"]
        saturation_min = self.hparams["saturation_min"]
        saturation_max = self.hparams["saturation_max"]
        value_min = self.hparams["value_min"]
        value_max = self.hparams["value_max"]

        mask = cv.inRange(domino, (hue_min, saturation_min, value_min), (hue_max, saturation_max, value_max))
        blur = cv.GaussianBlur(mask, (2 * guassian_blur_kernel + 1, 2 * guassian_blur_kernel + 1), 0)
        blur = cv.medianBlur(blur, 2 * median_blur_kernel + 1)
        _, thresh = cv.threshold(blur, threshold, 255, cv.THRESH_BINARY)
        kernel = np.ones((erode_kernel_size, erode_kernel_size), np.uint8)
        thresh = cv.erode(thresh, kernel, iterations=erode_iterations)
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        thresh = cv.dilate(thresh, kernel, iterations=dilate_iterations)

        result = cv.cvtColor(thresh, cv.COLOR_GRAY2BGR)
        # print(f"Thresh: {type(result)} {result.shape}")

        return result

if __name__ == "__main__":
    domino_cleaner = DominoCleaner("clean_domino2.pickle")
    root_path = "../../extracted_domino_pieces"
    save_path = "../../extracted_domino_pieces_clean"
    for file in os.listdir(root_path):
        if file.endswith(".jpg"):
            domino = cv.imread(os.path.join(root_path, file))
            domino = domino_cleaner.clean_domino(domino)
            cv.imwrite(os.path.join(save_path, file), domino)

    # domino = cv.imread("../../extracted_domino_pieces/1_1.jpg")
    # for i in range(0, 7):
        # for j in range(i, 7):
            # print(j, i)
            # domino = cv.imread("../../default_domino_pieces/{}_{}.jpg".format(j, i))
            # domino = domino_cleaner.clean_domino(domino)
            # cv.imwrite("../../default_domino_pieces_clean/{}_{}.jpg".format(j, i), domino)
            # cv.imshow("Domino", domino)
            # cv.waitKey(0)

    # domino = cv.imread("../../default_domino_pieces/3_2.jpg")
    # domino = domino_cleaner.clean_domino(domino)
    # cv.imshow("Domino", domino)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

