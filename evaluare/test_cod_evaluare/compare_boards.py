import numpy as np
import cv2 as cv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from board_extractor import BoardExtractor
from utils import get_horizontal_and_vertical_lines


def get_submatrix(image: np.ndarray, top_left: tuple, bottom_right: tuple) -> np.ndarray:
    return image[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :]

if __name__ == "__main__":
    board_extractor = BoardExtractor("config6.pickle", 900, 900)
    board_extractor.view_config()

    game_path = "../../antrenare/"

    start_board = "../../imagini_auxiliare/01.jpg"
    game_index = 1
    game_board_files = [file for file in os.listdir(game_path) if file.startswith(f"{game_index}_") and file.endswith(".jpg")]
    game_board_files = sorted(game_board_files, key=lambda x: int(x.split(".")[0].split("_")[1]))
    game_board_files = [os.path.join(game_path, file) for file in game_board_files]
    game_board_files = [start_board] + game_board_files


    game_boards = []
    for board_file in tqdm(game_board_files, desc="Extracting boards"):
        # print(board_file)
        image = cv.imread(board_file)
        curr_board = board_extractor.extract_board(image)
        game_boards.append(curr_board)


    # game_boards = [draw_grid(board) for board in game_boards]
    # for board in game_boards[1:]:   
        # diff = cv.absdiff(last_board, board)
        # cv.imshow("Diff", diff)
        # cv.waitKey(0)
        # last_board = board
    # cv.destroyAllWindows()

    step_size = 900 // 15
    last_board = game_boards[0] # this should be the start board

    for idx, board in tqdm(enumerate(game_boards[1:]), desc="Computing diff"):
        diff_board = cv.absdiff(board, last_board)
        # if idx + 1 < 10:
        #     str_idx = f"0{idx + 1}"
        # else:
        #     str_idx = f"{idx + 1}"
        # cv.imwrite(f"../../domino_extract/{game_index}_{str_idx}.jpg", diff_board)

        diff_values = []
        for i in range(0, 15):
            for j in range(0, 15):
                top_left = (i * step_size, j * step_size)
                bottom_right = ((i + 1) * step_size, (j + 1) * step_size)

                submatrix = get_submatrix(board, top_left, bottom_right)
                last_submatrix = get_submatrix(last_board, top_left, bottom_right)

                diff = cv.absdiff(submatrix, last_submatrix)
                
                diff_values.append((np.mean(diff), i, j))

                result = np.zeros((900, 900, 3), dtype=np.uint8)
                result[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1], :] = diff

                # cv.imshow("result", result)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

        diff_values = sorted(diff_values, key=lambda x: x[0], reverse=True)
        domino_piece = diff_values[:4]
        # print(domino_piece)

        img = board.copy()
        for piece in domino_piece:
            top_left = (piece[1] * step_size, piece[2] * step_size)
            bottom_right = ((piece[1] + 1) * step_size, (piece[2] + 1) * step_size)
            cv.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

        # cv.imshow("Board", img)
        # cv.waitKey(0)

        last_board = board
    # cv.destroyAllWindows()

    # plt.plot(diff_values)
    # plt.show()




