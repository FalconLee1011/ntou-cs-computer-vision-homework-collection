import tensorflow
import numpy as np
import cv2

from hw1_filter.operations.colorCord import colorCord


def applyFilter(videoPath):
    vid = cv2.VideoCapture(videoPath)

    while vid.isOpened():
        ret, frame = vid.read()
        if ret == True:
            frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb2 = colorCord().bgr2rgb(frame)
            heatMap = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)

            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame, 'raw', (15, 25), font, 1.15, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(rgb, 'BGR -> RGB', (15, 25), font, 1.15, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(rgb2, 'BGR -> RGB(NUMPY)', (15, 25), font, 1.15, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(heatMap, 'HEATMAP', (15, 25), font, 1.15, (255, 255, 255), 1, cv2.LINE_AA)

            row = np.vstack((frame, rgb))
            row2 = np.vstack((rgb2, heatMap))

            full = np.hstack((row, row2))

            cv2.imshow("result", full)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            break

    vid.release()
    cv2.destroyAllWindows()
