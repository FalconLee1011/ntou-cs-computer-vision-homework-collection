import argparse


def initARG():
    parser = argparse.ArgumentParser(
        prog="Computer Vision HW1 - Filters",
        description="No further description.",
    )
    parser.add_argument("--source", help="Input video")
    args = parser.parse_args()
    return args


def main():
    args = initARG()
    if args.source:
        from hw1_filter.CVfilter import applyFilter

        print(f"Source {args.source}")
        applyFilter(args.source)
    else:
        raise Exception("No video provided, please use --source <path-to-input-video>")


if __name__ == "__main__":
    main()
