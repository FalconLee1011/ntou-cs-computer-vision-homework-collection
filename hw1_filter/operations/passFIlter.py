import numpy as np
import tensorflow as tf
from cv2 import getGaussianKernel

from hw1_filter.utils.tfUtil import TFUtil


class PASSFilter:
    @staticmethod
    def high_pass(frame, freq):
        f = np.fft.fft2(frame)
        fshift = np.fft.fftshift(f)
        cy, cx = fshift.shape[0] / 2, fshift.shape[1] / 2
        h = np.arange(fshift.shape[0]).reshape((-1, 1)) - cy
        w = np.arange(fshift.shape[1]).reshape((1, -1)) - cx
        freq = freq ** 2
        fshift = np.where(h ** 2 + w ** 2 >= freq, fshift, 0)
        return fshift
