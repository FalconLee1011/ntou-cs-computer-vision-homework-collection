import argparse


def initARG():
    parser = argparse.ArgumentParser(
        prog="Computer Vision HW3 - Stitching",
        description="Stitching images or videos",
    )
    parser.add_argument(
        "--type", help="Input type [ video | frames ], default is video"
    )
    parser.add_argument(
        "--source",
        help="Input path, if frames are specified, it must be a path to a folder contains frames.",
    )
    parser.add_argument("--reverse", help="Reverse input video.")
    parser.add_argument(
        "--direction", help="Specify input video direction. [ ltr | rtl ]"
    )
    parser.add_argument(
        "--minmatch",
        help="Minium match points, if match points is less than specified number, it will not be stitched.",
    )
    # parser.add_argument("--output", help="Output video name")
    args = parser.parse_args()
    return args


def main():
    args = initARG()
    if args.source:
        from hw3_stitching.stitching import stitch

        print(f"Source {args.source}")
        stitch(args)
    else:
        print("No video provided, please use --source <path-to-input-video>")


if __name__ == "__main__":
    main()
