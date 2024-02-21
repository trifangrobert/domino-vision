import numpy as np
import cv2 as cv
import os
from typing import Tuple

class DominoClassifier:
    def __init__(self, labels_path: str) -> None:
        self.labels = {}
        for file in os.listdir(labels_path):
            if file.endswith(".jpg"):
                domino = cv.imread(os.path.join(labels_path, file))
                self.labels[file.split(".")[0]] = domino
        


    def classify_domino(self, domino: np.ndarray) -> Tuple[int, int]:
        width = domino.shape[1]
        height = domino.shape[0]

        if width > height:
            domino = cv.rotate(domino, cv.ROTATE_90_CLOCKWISE)

        
        mx = -np.inf
        mx_pos = None

        print("Normal")
        for label, domino_template in self.labels.items():
            corr = cv.matchTemplate(domino, domino_template, cv.TM_CCOEFF_NORMED) # might need to expand the domino a bit
            corr = np.max(corr) 
            print(f"{label}: {corr}")
            if corr > mx:
                mx = corr
                mx_pos = label

        print(f"Best normal values {mx} {mx_pos}")

        # cv.imshow("Domino", domino) 
        # cv.imshow("Best match", self.labels[mx_pos])
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        mx_rev = -np.inf
        mx_rev_pos = None
        domino = cv.rotate(domino, cv.ROTATE_180)

        print("Rotated")

        for label, domino_template in self.labels.items():
            corr = cv.matchTemplate(domino, domino_template, cv.TM_CCOEFF_NORMED) # might need to expand the domino a bit
            corr = np.max(corr) 
            print(f"{label}: {corr}")
            if corr > mx_rev:
                mx_rev = corr
                mx_rev_pos = label

        print(f"Best rotated values {mx_rev} {mx_rev_pos}")

        # cv.imshow("Domino", domino) 
        # cv.imshow("Best match", self.labels[mx_rev_pos])
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        if mx > mx_rev:
            x, y = mx_pos.split("_")
            x = int(x)
            y = int(y)
            return x, y
        else:
            x, y = mx_rev_pos.split("_")
            x = int(x)
            y = int(y)
            return y, x
            

        

if __name__ == "__main__":
    domino_classifier = DominoClassifier("../../default_domino_pieces_clean")
    
    
    for file in os.listdir("../../extracted_domino_pieces_clean"):
        if file.endswith(".jpg"):
            domino = cv.imread(os.path.join("../../extracted_domino_pieces_clean", file))
            cv.imshow("Domino", domino)
            cv.waitKey(0)
            domino_class = domino_classifier.classify_domino(domino)
            print(domino_class)
            # break

    cv.destroyAllWindows()
                