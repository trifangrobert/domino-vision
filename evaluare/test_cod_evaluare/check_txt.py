import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Checking results')
    parser.add_argument('-gt', '--gt_path', type=str, help='Path to the ground truth', required=True)
    parser.add_argument('-r', '--result_path', type=str, help='Path to the results', required=True)
    parser.add_argument('-gi', '--game_index', type=str, help='Game index')
    args = parser.parse_args()

    # if args.gt_path is None or args.result_path is None or args.game_index is None:
    #     print("Missing arguments")
    #     exit(1)

    gt_path = args.gt_path
    result_path = args.result_path
    game_index = args.game_index

    points_task1 = 0
    points_task2 = 0
    points_task3 = 0
    for file in tqdm(os.listdir(result_path), desc="Checking results"):
        if file.endswith(".txt") and (not args.game_index or (args.game_index and file.startswith(args.game_index))):
            result_file = file.split(".")[0]
            with open(os.path.join(gt_path, file), "r") as f:
                text = f.readlines()
                gt_pos1, gt_val1 = text[0].split()
                gt_pos2, gt_val2 = text[1].split()
                gt_score = text[2].split()[0]

            with open(os.path.join(result_path, file), "r") as f:
                text = f.readlines()
                pos1, val1 = text[0].split()
                pos2, val2 = text[1].split()
                score = text[2].split()[0]

            if gt_pos1 == pos1 and gt_pos2 == pos2:
                points_task1 += 0.05
            
            if gt_val1 == val1 and gt_val2 == val2:
                points_task2 += 0.02

            if gt_score == score:
                points_task3 += 0.02

    print(f"Task 1: {round(points_task1, 2)}/5.0")
    print(f"Task 2: {round(points_task2, 2)}/2.0")
    print(f"Task 3: {round(points_task3, 2)}/2.0")

    print(f"Total: {round(points_task1 + points_task2 + points_task3, 2)}/9.0")
