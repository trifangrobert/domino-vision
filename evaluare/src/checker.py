from typing import List
from tqdm import tqdm
import os

from domino import Domino

class Checker:
    def __init__(self, dominoes: List[Domino], answer_path: str) -> None:
        self.dominoes = dominoes
        self.answer_path = answer_path

    def check(self) -> None:
        correct = 0
        for domino in tqdm(self.dominoes, desc="Verifying dominoes"):
            answer_file = domino.domino_file.split(".")[0] + ".txt"
            with open(os.path.join(self.answer_path, answer_file), "r") as f:
                text = f.readlines()
                gt_pos1, gt_val1 = text[0].split()
                gt_pos2, gt_val2 = text[1].split()

            if gt_pos1 != domino.pos1 or gt_pos2 != domino.pos2 or gt_val1 != str(domino.val1) or gt_val2 != str(domino.val2):
                print(f"Domino {domino.domino_file} is invalid")
                continue
            else:
                correct += 1

        print(f"Correct: {correct}/{len(self.dominoes)}")

