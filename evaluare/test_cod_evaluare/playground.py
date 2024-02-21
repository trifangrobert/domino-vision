import numpy as np
import cv2 as cv
import os
from tqdm import tqdm

if __name__ == "__main__":
    domino_positions = [("1A", "2A"), ("1C", "2C"), ("1E", "2E"), ("1G", "2G"), ("1I", "2I"), ("1K", "2K"), ("1M", "2M"), ("4A", "5A"), ("4C", "5C"), ("4E", "5E"), ("4G", "5G"), ("4I", "5I"), ("4K", "5K"), ("7A", "8A"), ("7C", "8C"), ("7E", "8E"), ("7G", "8G"), ("7I", "8I"), ("10A", "11A"), ("10C", "11C"), ("10E", "11E"), ("10G", "11G"), ("13A", "14A"), ("13C", "14C"), ("13E", "14E"), ("10M", "11M"), ("10O", "11O"), ("13O", "14O")]
    domino_values = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (2, 2), (3, 2), (4, 2), (5, 2), (6, 2), (3, 3), (4, 3), (5, 3), (6, 3), (4, 4), (5, 4), (6, 4), (5, 5), (6, 5), (6, 6)]

    image = cv.imread("../../default_domino_pieces/02.jpg")

    image = cv.resize(image, (900, 900))

    for domino_pos, domino_value in zip(domino_positions, domino_values):
        top_left = domino_pos[0]
        bottom_right = domino_pos[1]

        # top_left and bottom_right will look like this: 8H 8I

        row_top_left = int(top_left[:-1]) - 1
        col_top_left = top_left[-1]

        row_bottom_right = int(bottom_right[:-1]) - 1
        col_bottom_right = bottom_right[-1]

        step_size = 900 // 15

        row_top_left = row_top_left * step_size
        col_top_left = (ord(col_top_left) - 65) * step_size

        row_bottom_right = (row_bottom_right + 1) * step_size
        col_bottom_right = ((ord(col_bottom_right) - 65) + 1) * step_size

        domino = image[row_top_left:row_bottom_right, col_top_left:col_bottom_right]

        cv.imshow("Domino", domino)
        cv.waitKey(0)
    cv.destroyAllWindows()
        
