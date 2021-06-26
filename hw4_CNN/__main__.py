import argparse


def initARG():
    parser = argparse.ArgumentParser(
        prog="Computer Vision HW4 - CNN Fruit Recognition",
    )
    parser.add_argument("--mode", help="Mode [ load | train ]")
    parser.add_argument("--model", help="Select model [ DenseModel | DenseBlockModel ]")
    parser.add_argument(
        "--modelpath", help="If mode is 'load', you must specify a model."
    )
    parser.add_argument("--trainpath", help="Datasets for training.")
    parser.add_argument("--testpath", help="Datasets for testing.")
    # parser.add_argument("--output", help="Output video name")
    args = parser.parse_args()
    return args


def main():
    args = initARG()
    print(args)
    if (
        args.mode is None
        and args.model is None
        and args.trainpath is None
        and args.testpath is None
    ):
        print("No mode, model, trainpath, testpath provided.")

    elif args.mode == "train":
        from hw4_CNN import fruits

        fruits.train_and_test(args.model, args.trainpath, args.testpath)

    elif args.mode == "load":
        from hw4_CNN import fruits

        fruits.loadmodel_and_test(
            args.model, args.modelpath, args.trainpath, args.testpath
        )

    else:
        print("No model, trainpath, testpath provided.")


if __name__ == "__main__":
    main()
