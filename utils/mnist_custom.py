import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img


def set_tr_val_samples_labels(meta_filename, val_size):
    column_names = ["filename", "class_label"]
    data_frame = pd.read_csv(meta_filename, header=None, names=column_names, sep="\s+")
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)
    split_idx = int(len(data_frame) * (1 - val_size))
    return (
        data_frame["filename"][:split_idx].tolist(),
        data_frame["class_label"][:split_idx].tolist(),
        data_frame["filename"][split_idx:].tolist(),
        data_frame["class_label"][split_idx:].tolist(),
    )


def set_test_samples_labels(meta_filename):
    sample_paths, class_labels = [], []
    column_names = ["filename", "class_label"]
    data_frame = pd.read_csv(meta_filename, header=None, names=column_names, sep="\s+")
    return data_frame["filename"].tolist(), data_frame["class_label"].tolist()


class MNISTDataloader(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def get_image(self, sample_path):
        img = Image.open("datasets/MNIST/raw/" + sample_path)
        return img

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size, dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = self.get_image(path)
            x[j] = img
        y = np.asarray(batch_target_img_paths)
        return x, y


if __name__ == "__main__":
    tr_sample_paths, tr_class_labels, val_sample_paths, val_class_labels = set_tr_val_samples_labels(
        "datasets/Rotated_MNIST/metadata/train_kfold.txt", 0.2
    )
    te_sample_paths, te_class_labels = set_test_samples_labels("datasets/Rotated_MNIST/metadata/test_kfold.txt")
    train_set = MNISTDataloader(32, (28, 28), tr_sample_paths, tr_class_labels)
    val_set = MNISTDataloader(32, (28, 28), val_sample_paths, val_class_labels)
    test_set = MNISTDataloader(32, (28, 28), te_sample_paths, te_class_labels)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )

    model.compile(
        optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"]
    )

    model.fit(train_set, validation_data=val_set, epochs=5)
    model.evaluate(test_set)
