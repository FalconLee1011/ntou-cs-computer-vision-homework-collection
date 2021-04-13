import cv2
from numpy import abs as npAbs
from numpy import fft as npFft
from numpy import float32 as npFloat32
from numpy import hstack as npHStack
from numpy import uint8 as npUnit8
from numpy import vstack as npVStack
from numpy import zeros as npZeros

from hw1_filter.operations.passFIlter import PASSFilter
from hw1_filter.operations.tfFilter import TFFilter


def applyFilter(videoPath):
    vid = cv2.VideoCapture(videoPath)

    while vid.isOpened():
        ret, frameRaw = vid.read()
        if ret == True:

            HPFRQ = 2
            font = cv2.FONT_HERSHEY_PLAIN

            # Raw, resize
            frame = cv2.resize(frameRaw, (0, 0), None, 0.25, 0.25)

            # Blank
            # cv2.putText(
            #     frame, "raw", (15, 25), font, 1.15, (255, 255, 255), 1, cv2.LINE_AA
            # )
            # height, width, channels = frame.shape
            # _blank = npZeros(frame.shape, npUnit8)

            # highPass
            highPass = PASSFilter.high_pass(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), HPFRQ
            )
            highPassFrame = npUnit8(npAbs(npFft.ifft2(npFft.fftshift(highPass))))
            highPassFrame = cv2.cvtColor(highPassFrame, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                highPassFrame,
                f"HIGHPASS frq@{HPFRQ}",
                (15, 25),
                font,
                1.15,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            rgb = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.putText(
                rgb,
                "RGB -> BGR (CV2)",
                (15, 25),
                font,
                1.15,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            heatMap = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)
            cv2.putText(
                heatMap,
                "HEATMAP (CV2)",
                (15, 25),
                font,
                1.15,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            tf = TFFilter().flip(frame)
            cv2.putText(
                tf,
                "TF Flip",
                (15, 25),
                font,
                1.15,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            row = npVStack((highPassFrame, rgb))
            row2 = npVStack((tf, heatMap))

            full = npHStack((row, row2))

            cv2.imshow("result", full)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        else:
            break

    vid.release()
    cv2.destroyAllWindows()
