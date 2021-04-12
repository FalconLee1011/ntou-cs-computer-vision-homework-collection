from pprint import pprint

import numpy as np
import tensorflow as tf
from cv2 import getGaborKernel


class colorCord:
    @staticmethod
    def tfRGB2BGR(frame: np.ndarray) -> np.ndarray:
        tfImage = tf.ragged.constant(frame)
        return tfImage.numpy()
