import cv2
import numpy as np
import tensorflow as tf


class CV2Util:
    @staticmethod
    def splitFrame(rawFrame: np.ndarray, HCuts: int = 1, WCuts: int = 1) -> list:
        cutFrames = list()
        h, w, c = rawFrame.shape
        print(rawFrame.shape)
        ch = h // HCuts
        cw = w // WCuts
        print(f"Splitting frame {w} * {h} to {cw} * {ch} ({WCuts * HCuts})")
        for WCut in range(WCuts):
            for HCut in range(HCuts):
                _rawFrame = np.copy(rawFrame)
                originW = cw * WCut
                originH = ch * HCut
                cropped = _rawFrame[originH : originH + ch, originW : originW + cw, :]
                # cv2.imshow("cut", cropped)
                # print(cropped.shape)
                # cv2.waitKey(0)
                cutFrames.append(cropped)
        return cutFrames
