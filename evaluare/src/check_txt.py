import argparse
import os
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Checking results')
    parser.add_argument('-gt', '--gt_path', type=str, help='Path to the ground truth', required=True)
    parser.add_argument('-r', '--result_path', type=str, help='Path to the results', required=True)
    parser.add_argument('-gi', '--game_index', nargs='+', type=int, help="Index(es) of the game(s) to check.", required=True)
    args = parser.parse_args()

    gt_path = args.gt_path
    result_path = args.result_path
    game_index = args.game_index        

    total_points = [0, 0, 0]
    total_limits = [1.0, 0.4, 0.4]
    games = 0

    for idx in args.game_index:        
        points = [0, 0, 0]
        limits = [1.0, 0.4, 0.4]
        found_files = 0
        for file in tqdm(os.listdir(result_path), desc="Checking results"):
            if file.endswith(".txt") and file.startswith(str(idx)):
                found_files += 1
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
                    points[0] += 0.05
                
                if gt_val1 == val1 and gt_val2 == val2:
                    points[1] += 0.02

                if gt_score == score:
                    points[2] += 0.02

        if found_files != 20:
            print(f"\nGame {idx} is incomplete. Found {found_files} files out of 20.\n")
            continue
        else:
            games += 1
            

        total_points[0] += points[0]
        total_points[1] += points[1]
        total_points[2] += points[2]

        print()
        print(f"Game {idx}")
        print(f"Task 1: {round(points[0], 2)}/{round(limits[0], 2)}")
        print(f"Task 2: {round(points[1], 2)}/{round(limits[1], 2)}")
        print(f"Task 3: {round(points[2], 2)}/{round(limits[2], 2)}")
        print()
            
    total_limits = [x * games for x in total_limits]
    score = [total_points[0] / total_limits[0], total_points[1] / total_limits[1], total_points[2] / total_limits[2]]
    print(f"Total:")
    for task in range(3):
        print(f"Task {task + 1} accuracy: {100 * round(score[task], 2)}% and points: {round(total_points[task], 2)}/{round(total_limits[task], 2)}")

