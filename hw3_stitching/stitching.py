from os import walk as os_walk
from os.path import isdir as os_path_isdir
from os.path import join as os_path_join

import cv2
import numpy as np

frame_num = 0
frame_count = 0
frames = list()
source_width = 0
source_height = 0

result = None
count = None
ones = None
cap = None

min_match = 0


def stitch(args):
    global min_match
    type_ = args.type
    source = args.source
    min_match = args.minmatch or 500
    min_match = int(min_match)
    if type_ == "video":
        if os_path_isdir(source):
            print("Source is not a video.")
            exit(-1)
        loadFramesFromVideo(source, args.reverse)
        _stitch_video(args.direction)
    elif type_ == "frames":
        if not os_path_isdir(source):
            print("Source is not a folder.")
            exit(-1)
        loadFramesFromFrames(source)
        _stitch_frames(args.direction)
    else:
        print(f"Unknown type {type_}")
        exit(-1)


def loadFramesFromVideo(path, reverse=False):
    global frames, frame_count, source_width, source_height, result, count, ones, cap
    cap = cv2.VideoCapture(path)
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    result = np.zeros((source_height // 4, source_width // 4 * 3, 3))
    count = np.zeros((source_height // 4, source_width // 4 * 3))
    ones = np.ones(((source_height // 4, source_width // 4)))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read the frame as array
    f = 0
    while f < frame_count:
        ret, frame = cap.read()
        if ret == False:
            break
        frames.append(frame)
        print(f"reading frame {f}")
        f += 1

    if reverse:
        frames.reverse()


def loadFramesFromFrames(path):
    global frames, frame_count, source_width, source_height, result, count, ones, cap
    fc = 0
    for root, dirs, files in os_walk(path):
        fc = len(files)
        files.sort()
        for file in files:
            fp = os_path_join(root, file)
            print(f"Reading {fp}")
            frames.append(cv2.imread(fp, cv2.IMREAD_COLOR))

    cap = frames[0]
    source_height, source_width, _ = cap.shape
    result = np.zeros((source_height // 4 * 3, source_width // 4 * 3, 3))
    count = np.zeros((source_height // 4 * 3, source_width // 4 * 3))
    ones = np.ones(((source_height // 4, source_width // 4)))
    frame_count = fc


def _stitch_video(direction="rtl"):
    global frames, frame_count, source_width, source_height, result, count, ones, cap

    last_frame = None
    last_kp = None
    last_dt = None

    def set_frame_number(x):
        global frame_num
        frame_num = x
        return

    cv2.namedWindow("Matching")
    cv2.createTrackbar("Frame number", "Matching", 0, frame_count - 1, set_frame_number)

    kpdetector = cv2.SIFT_create()
    # kpdetector = cv2.AKAZE_create()

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    for (frame_num, frame) in enumerate(frames):
        cv2.setTrackbarPos("frame no.", "Matching", frame_num)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp = kpdetector.detect(gray, None)
        dt = kpdetector.compute(gray, kp)[1]

        if frame_num == 0:
            T = np.eye(3)
            # T[0,2] = result.shape[1] -frame2.shape[1]
            T[0, 2] = 0 if (direction == "ltr") else result.shape[1] - frame.shape[1]
            T[1, 2] = 0
            result = cv2.warpPerspective(
                frame, T, (result.shape[1], result.shape[0])
            ).astype(np.float)
            t_count = cv2.warpPerspective(
                ones, T, (result.shape[1], result.shape[0])
            ).astype(np.float)
            count += t_count.astype(np.float)
            disp = result.copy()
            cv2.imshow("stitched image", disp.astype(np.uint8))

        else:
            matches = bf.match(dt, last_dt)
            print("{}, # of matches:{}".format(frame_num, len(matches)))

            matches = sorted(matches, key=lambda x: x.distance)

            src = []
            dst = []
            for m in matches:
                src.append(kp[m.queryIdx].pt + (1,))
                dst.append(last_kp[m.trainIdx].pt + (1,))

            src = np.array(src, dtype=np.float)
            dst = np.array(dst, dtype=np.float)

            # find a homography to map src to dst
            A, mask = cv2.findHomography(src, dst, cv2.RANSAC)

            # map to the first frame
            T = T.dot(A)
            warp_img = cv2.warpPerspective(
                frame, T, (result.shape[1], result.shape[0])
            ).astype(np.float)
            t_count = cv2.warpPerspective(
                ones, T, (result.shape[1], result.shape[0])
            ).astype(np.float)
            result += warp_img
            count += t_count.astype(np.float)

            t_count = count.copy()
            t_count[t_count == 0] = 1
            disp = result.copy()

            disp[:, :, 0] = result[:, :, 0] / t_count
            disp[:, :, 1] = result[:, :, 1] / t_count
            disp[:, :, 2] = result[:, :, 2] / t_count

            cv2.imshow("stitched image", disp.astype(np.uint8))

            try:
                cv2.imshow(
                    "Matching",
                    cv2.drawMatches(
                        frame,
                        last_kp,
                        last_frame,
                        kp,
                        matches[:15],
                        None,
                        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                    ),
                )
            except:
                print(f"Error displaying 'Matching' @ frame {frame_num}")

        last_frame = frame
        last_kp = kp
        last_dt = dt

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        # frame_num += 1

    cv2.waitKey()
    cap.release()
    cv2.destroyAllWindows()


def _stitch_frames(direction="rtl"):
    global frames, frame_count, source_width, source_height, result, count, ones, cap

    last_frame = None
    last_kp = None
    last_dt = None

    def set_frame_number(x):
        global frame_num
        frame_num = x
        return

    cv2.namedWindow("Matching")
    cv2.createTrackbar("Frame number", "Matching", 0, frame_count - 1, set_frame_number)

    kpdetector = cv2.SIFT_create()
    # kpdetector = cv2.AKAZE_create()

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    for (frame_num, frame) in enumerate(frames):
        cv2.setTrackbarPos("frame no.", "Matching", frame_num)
        # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)

        frame = cv2.resize(frame, (frame.shape[1] // 4, frame.shape[0] // 4))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp = kpdetector.detect(gray, None)
        dt = kpdetector.compute(gray, kp)[1]

        if frame_num == 0:
            T = np.eye(3)
            # T[0,2] = result.shape[1] -frame2.shape[1]
            T[0, 2] = result.shape[0] / 2
            T[1, 2] = result.shape[1] / 4
            result = cv2.warpPerspective(
                frame, T, (result.shape[1], result.shape[0])
            ).astype(np.float)
            t_count = cv2.warpPerspective(
                ones, T, (result.shape[1], result.shape[0])
            ).astype(np.float)
            count += t_count.astype(np.float)
            disp = result.copy()
            cv2.imshow("stitched image", disp.astype(np.uint8))

        else:
            matches = bf.match(dt, last_dt)
            print("{}, # of matches:{}".format(frame_num, len(matches)))
            if len(matches) < min_match:
                print(f"Match point is less than minimum requirement ({min_match}).")
                continue
            matches = sorted(matches, key=lambda x: x.distance)

            src = []
            dst = []
            for m in matches:
                src.append(kp[m.queryIdx].pt + (1,))
                dst.append(last_kp[m.trainIdx].pt + (1,))

            src = np.array(src, dtype=np.float)
            dst = np.array(dst, dtype=np.float)

            # find a homography to map src to dst
            A, mask = cv2.findHomography(src, dst, cv2.RANSAC)

            # map to the first frame
            T = T.dot(A)
            warp_img = cv2.warpPerspective(
                frame, T, (result.shape[1], result.shape[0])
            ).astype(np.float)
            t_count = cv2.warpPerspective(
                ones, T, (result.shape[1], result.shape[0])
            ).astype(np.float)
            result += warp_img
            count += t_count.astype(np.float)

            t_count = count.copy()
            t_count[t_count == 0] = 1
            disp = result.copy()

            disp[:, :, 0] = result[:, :, 0] / t_count
            disp[:, :, 1] = result[:, :, 1] / t_count
            disp[:, :, 2] = result[:, :, 2] / t_count

            cv2.imshow("stitched image", disp.astype(np.uint8))

            try:
                cv2.imshow(
                    "Matching",
                    cv2.drawMatches(
                        frame,
                        last_kp,
                        last_frame,
                        kp,
                        matches[:15],
                        None,
                        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                    ),
                )
            except:
                print(f"Error displaying 'Matching' @ frame {frame_num}")

        last_frame = frame
        last_kp = kp
        last_dt = dt

        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        # frame_num += 1

    cv2.waitKey()
    cap.release()
    cv2.destroyAllWindows()
