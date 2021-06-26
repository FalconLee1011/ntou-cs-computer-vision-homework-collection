from os import mkdir
from time import time

import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input as kerasInput, Model as kerasModel
from tensorflow.keras import layers as kerasLayers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DensenetBlockModel:
    def __init__(self, training_path, test_path, height, width, channel) -> None:
        print(f'Using "{self.__class__.__name__}"')

        self.height = height
        self.width = width
        self.channel = channel
        self.history = {
            "train": None,
            "test": None,
        }

        self.train_data_generator = ImageDataGenerator(
            horizontal_flip=True,
            preprocessing_function=tf.keras.applications.densenet.preprocess_input,
        )
        self.valid_data_generator = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.densenet.preprocess_input
        )
        self.test_data_generator = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.densenet.preprocess_input
        )

        self.train_generator = self.train_data_generator.flow_from_directory(
            training_path,
            target_size=(self.height, self.width),
            shuffle=True,
            batch_size=32,
            class_mode="sparse",
        )
        self.valid_generator = self.valid_data_generator.flow_from_directory(
            training_path,
            target_size=(self.height, self.width),
            shuffle=True,
            batch_size=32,
            class_mode="sparse",
        )
        self.test_generator = self.test_data_generator.flow_from_directory(
            test_path,
            target_size=(self.height, self.width),
            shuffle=True,
            batch_size=32,
            class_mode="sparse",
        )

    def Conv2D_BN(self, inputs, growth_rate):
        nfilter = growth_rate * 4
        outputs = kerasLayers.Activation("relu")(inputs)
        outputs = kerasLayers.BatchNormalization()(outputs)
        outputs = kerasLayers.Conv2D(
            filters=nfilter, kernel_size=(1, 1), padding="same", strides=1
        )(outputs)
        outputs = kerasLayers.Activation("relu")(outputs)
        outputs = kerasLayers.BatchNormalization()(outputs)
        outputs = kerasLayers.Conv2D(
            filters=growth_rate, kernel_size=(3, 3), padding="same", strides=1
        )(outputs)
        return outputs

    def dense_block(self, inputs, growth_rate, n_filter, layers):
        concat = inputs
        for _ in range(layers):
            outputs = self.Conv2D_BN(concat, growth_rate)
            concat = kerasLayers.concatenate([outputs, concat])
            n_filter += growth_rate
        return concat, n_filter

    def setup_model(self, nfilter=32, growth_rate=32):
        inputs = kerasInput(shape=(self.height, self.height, self.channel))
        x = kerasLayers.Conv2D(
            filters=nfilter,
            kernel_size=(7, 7),
            padding="same",
            strides=2,
            activation="relu",
        )(inputs)
        x = kerasLayers.BatchNormalization()(x)
        x = kerasLayers.MaxPooling2D(pool_size=2)(x)

        x, nfilter = self.dense_block(x, growth_rate, nfilter, 6)

        x = kerasLayers.GlobalAveragePooling2D()(x)
        x = kerasLayers.Dense(self.train_generator.num_classes, activation="softmax")(x)
        self.model = kerasModel(inputs=inputs, outputs=x)
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="Adam",
            metrics=["accuracy"],
        )

        return self.model

    def train(self, epochs):
        print(f"Training {self.__class__.__name__}")
        self.setup_model(self.train_generator.num_classes)
        self.model.summary()
        self.history["train"] = self.model.fit(
            self.train_generator, epochs=epochs, validation_data=self.valid_generator
        )
        print(f"Training of {self.__class__.__name__} is completed.")

    def test(self, summary=True, model=None):
        if model is not None:
            self.model = load_model(model)
        self.history["test"] = self.model.evaluate(self.test_generator)
        self.save_model()
        if summary:
            self.summary()

    def summary(self):
        print("\n\n")
        print("-------- Test --------")
        print(f"Loss : {self.history['test'][0]} Accuracy : {self.history['test'][1]}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history["train"].history["loss"], label="train_loss")
        plt.plot(self.history["train"].history["val_loss"], label="val_loss")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid(True)
        plt.show()

        plt.subplot(1, 2, 2)
        plt.plot(self.history["train"].history["accuracy"], label="train_accuracy")
        plt.plot(self.history["train"].history["val_accuracy"], label="val_accuracy")
        plt.legend()
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.grid(True)

        plt.show()

    def save_model(self):
        try:
            mkdir("cache")
        except:
            pass
        path = f"cache/{self.__class__.__name__}-{time()}"
        self.model.save(path)
        print(f"Model has been saved to {path}")
