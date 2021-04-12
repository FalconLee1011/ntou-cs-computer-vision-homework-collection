from numpy import (
    vstack as npVStack,
    hstack as npHStack,
    zeros as npZeros,
    uint8 as npUnit8,
)
import cv2


def applyFilter(videoPath):
    vid = cv2.VideoCapture(videoPath)

    while vid.isOpened():
        ret, frameRaw = vid.read()
        if ret == True:
            frame = cv2.resize(frameRaw, (0, 0), None, 0.25, 0.25)
            # height, width, channels = frame.shape
            _blank = npZeros(frame.shape, npUnit8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            heatMap = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)

            # tf = cv2.resize(tf, (0, 0), None, 0.25, 0.25)

            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(
                frame, "raw", (15, 25), font, 1.15, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                rgb, "RGB -> BGR", (15, 25), font, 1.15, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.putText(
                _blank, "TODO", (15, 25), font, 1.15, (255, 255, 255), 1, cv2.LINE_AA
            )
            # cv2.putText(
            #     tf, 'TF', (15, 25), font, 1.15, (255, 255, 255), 1, cv2.LINE_AA
            # )
            cv2.putText(
                heatMap,
                "HEATMAP",
                (15, 25),
                font,
                1.15,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            row = npVStack((frame, rgb))
            row2 = npVStack((_blank, heatMap))

            full = npHStack((row, row2))

            cv2.imshow("result", full)
            # cv2.imshow("tf", tf)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            break

    vid.release()
    cv2.destroyAllWindows()
