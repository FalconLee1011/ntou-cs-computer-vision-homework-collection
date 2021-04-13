import numpy as np
import tensorflow as tf
from cv2 import getGaussianKernel

from hw1_filter.utils.tfUtil import TFUtil


class TFFilter:
    @staticmethod
    def flip(frame: np.ndarray) -> np.ndarray:
        flipped = tf.image.flip_up_down(frame)
        return TFUtil.tfi2npi(flipped)
