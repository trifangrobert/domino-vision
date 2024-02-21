import numpy as np
import cv2 as cv
import argparse
import pickle
import os
from tqdm import tqdm

from board_extractor import BoardExtractor
from domino_extractor import DominoExtractor
from domino_cleaner import DominoCleaner
from domino_classifier import DominoClassifier
from domino import Domino
from checker import Checker

from utils import int_to_char

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning parameters')
    parser.add_argument('-b', '--board_config', type=str, help='Load extract board config file')
    parser.add_argument('-d', '--domino_config', type=str, help='Load extract dominoes config file')
    parser.add_argument('-c', '--clean_config', type=str, help='Load clean domino config file')
    parser.add_argument('-p', '--path', type=str, help='Path to the game images')
    parser.add_argument('-g', '--game_index', type=str, help='Game index')
    args = parser.parse_args()

    print(args)
    if args.board_config is None or args.domino_config is None or args.path is None:
        print("Missing arguments")
        exit(1)

    game_path = args.path
    game_index = args.game_index

    start_board = "../../imagini_auxiliare/01.jpg"
    game_board_files = [file for file in os.listdir(game_path) if file.startswith(f"{game_index}_") and file.endswith(".jpg")]
    game_board_files = sorted(game_board_files, key=lambda x: int(x.split(".")[0].split("_")[1]))
    game_board_files = [os.path.join(game_path, file) for file in game_board_files]
    game_board_files = [start_board] + game_board_files

    expand_board = 5
    expand_piece = 3

    board_extractor = BoardExtractor(args.board_config, 900, 900, expand_board)
    game_boards = []
    game_boards_expanded = []
    for board_file in tqdm(game_board_files, desc="Extracting boards"):
        image = cv.imread(board_file)
        curr_board, expanded_curr_board = board_extractor.extract_board(image)
        game_boards.append(curr_board)
        game_boards_expanded.append(expanded_curr_board)
    #     cv.imshow("Board", curr_board)
    #     cv.imshow("Board expanded", expanded_curr_board)
    #     cv.waitKey(0)
    # cv.destroyAllWindows()

    domino_configs = []
    domino_extractor = DominoExtractor(args.domino_config, width=900, height=900, offset=expand_board, expand_size=expand_piece)
    for board in tqdm(game_boards, desc="Extracting dominoes"):
        dominoes = domino_extractor.extract_dominoes(board)
        domino_configs.append(dominoes)

    domino_cleaner = DominoCleaner(args.clean_config)
    domino_classifier = DominoClassifier()

    dominoes_for_check = []
    # print(domino_configs)
    last_domino_config = domino_configs[0]
    valid_game = True
    for idx, domino_config in enumerate(domino_configs[1:]):
        str_idx = str(idx + 1)
        if len(str_idx) == 1:
            str_idx = "0" + str_idx
        diff = domino_config - last_domino_config
        cnt_diff = np.sum(diff) # this should be 2
        if cnt_diff != 2:
            valid_game = False
            print(f"Board {idx + 1} is invalid")
        
        # print(diff)
        domino_positions = np.where(diff == 1)
        cols, rows = domino_positions
        domino_positions = list(zip(rows, cols))
        # print(domino_positions)
        pos0 = str(domino_positions[0][0] + 1) + int_to_char(domino_positions[0][1])
        pos1 = str(domino_positions[1][0] + 1) + int_to_char(domino_positions[1][1])

        # domino_piece = domino_extractor.extract_domino(game_boards[idx + 1], [pos0, pos1])
        # print(type(game_boards_expanded[idx + 1]), game_boards_expanded[idx + 1].shape)

        domino_piece = domino_extractor.extract_domino(game_boards_expanded[idx + 1], [pos0, pos1])
        # print(type(domino_piece), domino_piece.shape)

        # cv.imwrite(f"../../extracted_domino_pieces/{game_index}_{str_idx}.jpg", domino_piece)
        # cv.imshow("Domino", domino_piece)
        # cv.waitKey(0)

        domino_piece_clean = domino_cleaner.clean_domino(domino_piece)
        # print(type(domino_piece_clean), domino_piece_clean.shape)

        # cv.imwrite(f"../../extracted_domino_pieces_clean/{game_index}_{str_idx}.jpg", domino_piece)
        # cv.imshow("Domino", domino_piece)
        # cv.waitKey(0)

        domino_class = domino_classifier.classify_domino(domino_piece_clean)

        dominoes_for_check.append(Domino(f"{game_index}_{str_idx}.jpg", pos0, pos1, domino_class[0], domino_class[1]))
        
        print(f"Domino {idx + 1} is {pos0} {pos1} {domino_class}")
        last_domino_config = domino_config

    # print(dominoes_for_check)

    with open(f"dominoes_for_check_{game_index}.pkl", "wb") as f:
        pickle.dump(dominoes_for_check, f)

    domino_checker = Checker(dominoes_for_check, args.path)
    domino_checker.check()
    # cv.destroyAllWindows()


    
