from pprint import pprint


import numpy as np
import cv2

class colorCord:

    @staticmethod
    def bgr2rgb(frame: np.ndarray) -> np.ndarray:
        _frame = np.ndarray()
        for ch in frame:
            for c in ch:
                c = [0, 0, 0]
        return frame