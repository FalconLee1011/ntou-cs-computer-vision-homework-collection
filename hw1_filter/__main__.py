import argparse


def initARG():
    parser = argparse.ArgumentParser(
        prog="Computer Vision HW1 - Filters",
        description="Applying 4 filters to video frames",
    )
    parser.add_argument("--source", help="Input video")
    parser.add_argument("--output", help="Output video name")
    args = parser.parse_args()
    return args


def main():
    args = initARG()
    if args.source:
        from hw1_filter.CVfilter import applyFilter

        print(f"Source {args.source}")
        applyFilter(args.source, True if args.output else False, args.output)
    else:
        print("No video provided, please use --source <path-to-input-video>")


if __name__ == "__main__":
    main()
