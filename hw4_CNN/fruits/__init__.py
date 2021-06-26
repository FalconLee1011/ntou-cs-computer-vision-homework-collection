from tensorflow.python.ops.gen_math_ops import mod

from .models.DensenetModel import DensenetModel
from .models.DenseBlockModel import DensenetBlockModel

MODELS = {
    "DensenetBlockModel": DensenetBlockModel,
    "DensenetModel": DensenetModel,
}


def train_and_test(
    model,
    train_path,
    test_path,
    epochs=3,
    image_heignt=100,
    image_width=100,
    image_channel=3,
):
    model = MODELS[model](
        train_path, test_path, image_heignt, image_width, image_channel
    )
    model.train(epochs)
    model.test()


def loadmodel_and_test(
    model,
    model_path,
    train_path,
    test_path,
    epochs=3,
    image_heignt=100,
    image_width=100,
    image_channel=3,
):
    model = MODELS[model](
        train_path, test_path, image_heignt, image_width, image_channel
    )
    # model.train(epochs)
    model.test(summary=False, model=model_path)
