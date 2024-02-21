import numpy as np
import cv2 as cv
import os
from typing import Tuple

class DominoClassifier:
    def __init__(self) -> None:
        self.minDist = 10
        self.param1 = 100
        self.param2 = 10
        self.minRadius = 5
        self.maxRadius = 8
        self.dp = 1
        self.blur_kernel = 5
        self.circle_threshold = 0.5

    def find_circles(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, self.dp, self.minDist, param1=self.param1, param2=self.param2, minRadius=self.minRadius, maxRadius=self.maxRadius)

        valid_circles = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                # count the number of black pixles in the circle
                black_count = 0
                cnt_pixels = 0
                for i in range(x-r, x+r):
                    for j in range(y-r, y+r):
                        if (i-x)**2 + (j-y)**2 <= r**2 and i >= 0 and j >= 0 and i < img.shape[1] and j < img.shape[0]:
                            if gray[j][i] == 0:
                                black_count += 1
                            cnt_pixels += 1

                if black_count/cnt_pixels > self.circle_threshold:
                    valid_circles.append((x, y, r))
        return valid_circles


    def classify_domino(self, domino: np.ndarray) -> Tuple[int, int]:
        cnt_0 = 0
        cnt_1 = 0
        circles = self.find_circles(domino)

        # img_circle = domino.copy()    
        # for circle in circles:
        #     cv.circle(img_circle, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)

        # print(domino.shape)
        # print(f"There are {len(circles)}: {circles}")
        if domino.shape[0] < domino.shape[1]:
            # draw a vertical line in the middle
            # cv.line(img_circle, (img.shape[1]//2, 0), (img.shape[1]//2, img.shape[0]), (0, 255, 0), 2)
            
            for circle in circles:
                if circle[0] < domino.shape[1]//2:
                    cnt_0 += 1
                else:
                    cnt_1 += 1

        else:
            # draw a horizontal line in the middle
            # cv.line(img_circle, (0, domino.shape[0]//2), (domino.shape[1], domino.shape[0]//2), (0, 255, 0), 2)

            for circle in circles:
                if circle[1] < domino.shape[0]//2:
                    cnt_0 += 1
                else:
                    cnt_1 += 1
        
        return (cnt_0, cnt_1)
