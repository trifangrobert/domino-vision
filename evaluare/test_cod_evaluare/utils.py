import cv2 as cv
import numpy as np

def apply_perpective_transform(image: np.ndarray, list_of_corners: list) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = list_of_corners

    width = 900
    height = 900

    # width = 600
    # height = 600

    puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")

    M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)

    result = cv.warpPerspective(image, M, (width, height))
    result = cv.cvtColor(result, cv.COLOR_HSV2BGR)

    return result


def get_horizontal_and_vertical_lines(width: int = 900, height: int = 900) -> tuple:
    horizontal_lines = []
    step_width = width // 15
    for i in range(0, 16):
        horizontal_lines.append([(0, i * step_width), (width - 1, i * step_width)])

    vertical_lines = []
    step_height = height // 15
    for i in range(0, 16):
        vertical_lines.append([(i * step_height, 0), (i * step_height, height - 1)])

    return horizontal_lines, vertical_lines


def draw_grid(image: np.ndarray) -> np.ndarray:
    horizontal_lines, vertical_lines = get_horizontal_and_vertical_lines()

    for line in horizontal_lines:
        cv.line(image, line[0], line[1], (0, 0, 255), 2)

    for line in vertical_lines:
        cv.line(image, line[0], line[1], (0, 0, 255), 2)

    return image

def get_corners(contours: list) -> tuple:
    max_area = 0
    top_left = None
    bottom_right = None
    top_right = None
    bottom_left = None 

    for i in range(len(contours)):
        if(len(contours[i]) >3):
            possible_top_left = None
            possible_bottom_right = None
            for point in contours[i].squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                    possible_bottom_right = point

            diff = np.diff(contours[i].squeeze(), axis = 1)
            possible_top_right = contours[i].squeeze()[np.argmin(diff)]
            possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
            if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                top_left = possible_top_left
                bottom_right = possible_bottom_right
                top_right = possible_top_right
                bottom_left = possible_bottom_left

    
    return top_left, top_right, bottom_right, bottom_left, max_area



def draw_corners(image: np.ndarray, list_of_corners: list) -> np.ndarray:
    top_left, top_right, bottom_right, bottom_left = list_of_corners

    image_copy = cv.cvtColor(image.copy(), cv.COLOR_HSV2BGR)
    cv.circle(image_copy,tuple(top_left),3,(0,0,255),-1)
    cv.circle(image_copy,tuple(top_right),3,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_left),3,(0,0,255),-1)
    cv.circle(image_copy,tuple(bottom_right),3,(0,0,255),-1)

    return image_copy 

def int_to_char(x: int) -> str:
    return chr(x + 65)

def char_to_int(x: str) -> int:
    return ord(x) - 65

def convert_from_domino_pos_to_board_pos(domino_pos: str) -> str:
    row = int(domino_pos[:-1]) - 1
    col = char_to_int(domino_pos[-1])
    return row, col

def convert_from_board_pos_to_domino_pos(board_pos: str) -> str:
    row = str(board_pos[0] + 1)
    col = int_to_char(board_pos[1])
    return row + col