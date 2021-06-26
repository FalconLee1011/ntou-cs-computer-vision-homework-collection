from os import mkdir
from time import time

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DensenetModel:
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

    def setup_model(self, class_count):
        self.model = Sequential(name="fruit_retailer")
        pretrain = tf.keras.applications.DenseNet121(
            include_top=False, input_shape=(self.height, self.width, self.channel)
        )
        self.model.add(pretrain)
        self.model.add(tf.keras.layers.GlobalAveragePooling2D())
        self.model.add(Dense(class_count, activation="relu"))
        self.model.add(Dense(class_count, activation="softmax"))
        pretrain.trainable = False
        self.model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer="Adam",
            metrics=["accuracy"],
        )

        return self.model

    def train(self, epochs):
        print(f"Training {self.__class__.__name__}")
        self.setup_model(self.train_generator.num_classes)
        self.history["train"] = self.model.fit(
            self.train_generator, epochs=epochs, validation_data=self.valid_generator
        )
        print(f"Training of {self.__class__.__name__} is completed.")
        self.model.summary()

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
        import matplotlib.pyplot as plt

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
