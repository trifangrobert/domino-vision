import numpy as np
from typing import List, Tuple
import cv2 as cv
import pickle
import os
from tqdm import tqdm
import argparse

from board_extractor import BoardExtractor
from domino_extractor import DominoExtractor
from domino_cleaner import DominoCleaner
from domino_classifier import DominoClassifier

from domino import Domino
from checker import Checker
from utils import convert_from_domino_pos_to_board_pos, convert_from_board_pos_to_domino_pos


class Game:
    def __init__(self, game_path: str, game_index: int, board_config: str = 'board_config.pickle', extract_domino_config: str = 'extract_domino_config.pickle', clean_domino_config: str = 'clean_domino_config3.pickle') -> None:
        self.game_path = game_path
        self.game_index = game_index
        self.board_config = board_config
        self.extract_domino_config = extract_domino_config
        self.clean_domino_config = clean_domino_config

        self._init_board()
        self._build_game()

    def _init_board(self) -> None:
        self.star_positions = []
        self.star_positions.append(self._get_star_positions([(0, 0)])) 
        self.star_positions.append(self._get_star_positions([(0, 3), (3, 0), (1, 5), (5, 1)]))
        self.star_positions.append(self._get_star_positions([(0, 7), (7, 0), (1, 2), (2, 1), (3, 3)]))
        self.star_positions.append(self._get_star_positions([(2, 4), (4, 2), (3, 5), (5, 3)]))
        self.star_positions.append(self._get_star_positions([(4, 4), (4, 6), (6, 4), (5, 5)]))
        self.star_values = [5, 4, 3, 2, 1]

        self.board = np.zeros((15, 15), dtype=np.int8)
        for pos, val in zip(self.star_positions, self.star_values):
            for (x, y) in pos:
                self.board[x, y] = val    

        self.roadmap = [-1, 1, 2, 3, 4, 5, 6, 0, 2, 5, 3, 4, 6, 2, 2, 0, 3, 5, 4, 1, 6, 2, 4, 5, 5, 0, 6, 3, 4, 2, 0, 1, 5, 1, 3, 4, 4, 4, 5, 0, 6, 3, 5, 4, 1, 3, 2, 0, 0, 1, 1, 2, 3, 6, 3, 5, 2, 1, 0, 6, 6, 5, 2, 1, 2, 5, 0, 3, 3, 5, 0, 6, 1, 4, 0, 6, 3, 5, 1, 4, 2, 6, 2, 3, 1, 6, 5, 6, 2, 0, 4, 0, 1, 6, 4, 4, 1, 6, 6, 3, 0]
        self.roadmap = np.array(self.roadmap, dtype=np.int8)

        self.pos_players = [0, 0]
        self.player_moves = []
        player_moves_file = os.path.join(self.game_path, f"{self.game_index}_mutari.txt")
        with open(player_moves_file, 'r') as f:
            lines = f.readlines()
            for idx in range(20):
                line = lines[idx].strip()
                file_name = line.split()[0]
                player_turn = int(line.split()[1][-1])
                self.player_moves.append((file_name, player_turn))
    

    def _extract_board(self) -> None:
        game_board_files = [file for file in os.listdir(self.game_path) if file.startswith(f"{self.game_index}_") and file.endswith(".jpg")]
        game_board_files = sorted(game_board_files, key=lambda x: int(x.split(".")[0].split("_")[1]))
        game_board_files = [os.path.join(self.game_path, file) for file in game_board_files]

        self.expand_board = 5

        self.board_extractor = BoardExtractor(self.board_config, 900, 900, self.expand_board)
        self.game_boards = []
        self.game_boards_expanded = []
        for board_file in tqdm(game_board_files, desc="Extracting boards"):
            image = cv.imread(board_file)
            curr_board, expanded_curr_board = self.board_extractor.extract_board(image)
            self.game_boards.append(curr_board)
            self.game_boards_expanded.append(expanded_curr_board)
        print("Done extracting boards")


    def _extract_all_dominoes(self) -> None: # this function extracts all the dominoes from the game
        self.expand_piece = 3

        self.domino_configs = []
        empty_board = np.zeros((15, 15), dtype=np.uint8)
        self.domino_configs.append(empty_board)
        self.domino_extractor = DominoExtractor(self.extract_domino_config, width=900, height=900, offset=self.expand_board, expand_size=self.expand_piece)
        for board in tqdm(self.game_boards, desc="Extracting dominoes"):
            dominoes = self.domino_extractor.extract_dominoes(board)
            self.domino_configs.append(dominoes)
        print("Done extracting dominoes")


    def _detect_added_dominoes(self) -> None: # this function detects, cleans and classifies the added dominoes 
        domino_cleaner = DominoCleaner(self.clean_domino_config)
        domino_classifier = DominoClassifier()

        last_domino_config = self.domino_configs[0]
        self.dominoes = []
        print("Detecting dominoes")
        for idx, domino_config in enumerate(self.domino_configs[1:]):
            str_idx = str(idx + 1)
            if len(str_idx) == 1:
                str_idx = "0" + str_idx
            diff = domino_config - last_domino_config
            cnt_diff = np.sum(diff) # this should be 2
            if cnt_diff != 2:
                print(f"Board {idx + 1} is invalid") # this should never happen
            
            domino_positions = np.where(diff == 1)
            cols, rows = domino_positions
            domino_positions = list(zip(rows, cols))
            
            pos0 = convert_from_board_pos_to_domino_pos(domino_positions[0])
            pos1 = convert_from_board_pos_to_domino_pos(domino_positions[1])

            domino_piece = self.domino_extractor.extract_domino(self.game_boards_expanded[idx], [pos0, pos1])
            domino_piece_cleaned = domino_cleaner.clean_domino(domino_piece)
            domino_class = domino_classifier.classify_domino(domino_piece_cleaned)

            self.dominoes.append(Domino(f"{self.game_index}_{str_idx}.jpg", pos0, pos1, domino_class[0], domino_class[1]))
            last_domino_config = domino_config
        
        print("Done detecting added dominoes")

    def check_game(self):
        # check if there are available ground truth files for the dominoes
        can_check = True
        for domino in self.dominoes:
            answer_file = domino.domino_file.split(".")[0] + ".txt"
            if not os.path.exists(os.path.join(self.game_path, answer_file)):
                can_check = False
                break

        if not can_check:
            print("Cannot check game, no ground truth files available")
            return
        
        checker = Checker(self.dominoes, self.game_path)
        checker.check()


    def _build_game(self):
        self._extract_board()
        self._extract_all_dominoes()
        self._detect_added_dominoes()


    def _get_star_positions(self, positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        star_positions = []
        for (x, y) in positions:
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
                new_x = x if dx == 0 else 14 * dx - x
                new_y = y if dy == 0 else 14 * dy - y
                star_positions.append((new_x, new_y))
        return star_positions
        
    def play_game(self, save_game_path: str = None) -> None:
        if save_game_path:
            print(f"Saving moves to {save_game_path}")
        
        for domino, player_move in zip(self.dominoes, self.player_moves):
            player_turn = player_move[1] - 1

            star_value = None # it should update only once per domino
            pos = convert_from_domino_pos_to_board_pos(domino.pos1)
            if self.board[pos[0], pos[1]] != 0:
                star_value = self.board[pos[0], pos[1]]
            pos = convert_from_domino_pos_to_board_pos(domino.pos2)
            if self.board[pos[0], pos[1]] != 0:
                star_value = self.board[pos[0], pos[1]]
            
            # if domino is placed on a star
            win_value = [0, 0]
            if star_value is not None:
                val = star_value
                if domino.val1 == domino.val2:
                    val *= 2
                win_value[player_turn] += val
            
            # if domino values is equal to the position on the roadmap
            if domino.val1 == self.roadmap[self.pos_players[0]] or domino.val2 == self.roadmap[self.pos_players[0]]:
                win_value[0] += 3
            
            if domino.val1 == self.roadmap[self.pos_players[1]] or domino.val2 == self.roadmap[self.pos_players[1]]:
                win_value[1] += 3
            
            for i in range(len(self.pos_players)):
                self.pos_players[i] += win_value[i]

            if save_game_path is not None:
                txt_file = domino.domino_file.split(".")[0] + ".txt"            
                save_path = os.path.join(save_game_path, txt_file)
                with open(save_path, 'w') as f:
                    f.write(f"{domino.pos1} {domino.val1}\n")
                    f.write(f"{domino.pos2} {domino.val2}\n")
                    f.write(f"{win_value[player_turn]}\n")
    
                 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Playing Double Double Dominoes')
    parser.add_argument('-p', "--game_path", type=str, required=True, help='Path to the game images')
    parser.add_argument('-g', "--game_index", type=int, required=True, help='Game index')
    parser.add_argument('-b', "--board_config", type=str, default='board_config.pickle', help='Load extract board config file')
    parser.add_argument('-d', "--extract_domino_config", type=str, default='extract_domino_config.pickle', help='Load extract dominoes config file')
    parser.add_argument('-c', "--clean_domino_config", type=str, default='clean_domino_config.pickle', help='Load clean domino config file')
    parser.add_argument('-t', "--test", action='store_true', help='Check game correctness')
    parser.add_argument('-s', "--save_game_path", type=str, default=None, help='Save game moves in the specified path')
    args = parser.parse_args()

    game = Game(args.game_path, args.game_index, args.board_config, args.extract_domino_config, args.clean_domino_config)
    game.play_game(args.save_game_path)
    if args.test:
        game.check_game()