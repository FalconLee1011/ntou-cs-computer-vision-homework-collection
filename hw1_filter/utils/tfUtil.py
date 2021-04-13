import numpy as np
import tensorflow as tf
from cv2 import getGaussianKernel


class TFUtil:
    @staticmethod
    def tfi2npi(frame: tf.image) -> np.ndarray:
        return frame.numpy().astype(np.uint8)

    @staticmethod
    def npi2tfi(
        frame: np.ndarray, reszie: float = 0.25, to_type: type = np.float32
    ) -> tf.image:

        gkernel = getGaussianKernel(9, 9.0)
        gfilter2d = np.dot(gkernel, gkernel.T)

        filters = np.zeros((9, 9, 3, 3))
        for i in range(3):
            filters[:, :, i, i] = gfilter2d

        _frame = frame.copy().astype(to_type)
        _frame = _frame[np.newaxis, ...]

        _frame2 = tf.squeeze(
            tf.image.resize(
                tf.nn.conv2d(_frame, filters, 1, "SAME"),
                (_frame.shape[0] * reszie, _frame.shape[1] * reszie),
            )
        )

        return _frame2
