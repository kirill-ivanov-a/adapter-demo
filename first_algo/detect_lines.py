import numpy as np

from PIL import Image


def detect_lines(image: Image):
    # image - RGB
    # detect(image) here
    x1_1, y1_1, x1_2, y1_2 = 0, 1, 0, 1
    x2_1, y2_1, x2_2, y2_2 = 5, 71, 11, 13
    # line = [[x_start, y_start], [x_end, y_end]]
    return np.array([[[x1_1, y1_1], [x1_2, y1_2]], [[x1_1, y1_1], [x1_2, y1_2]]])
