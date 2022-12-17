import os

import cv2
import imutils
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds


def make_dataset(dataset_name):
    ds = tfds.load(name="cifar10_1/" + dataset_name, split="test")
    numpy_items = tfds.as_numpy(ds)

    paths = []
    labels = []

    for index, item in enumerate(numpy_items):
        directory = "datasets/CIFAR-10/cifar10_1/" + dataset_name + "/" + str(item["label"])
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = str(item["label"]) + "/te_image_" + str(index) + ".png"
        cv2.imwrite("datasets/CIFAR-10/cifar10_1/" + dataset_name + "/" + path, item["image"])
        paths.append(path)
        labels.append(item["label"])

    return paths, labels


paths, labels = make_dataset("v6")
meta_files = pd.DataFrame({"path": paths, "label": labels})
meta_files.to_csv(
    "datasets/CIFAR-10/cifar10_1/metadata/test_v6_kfold.txt",
    header=None,
    sep=" ",
    encoding="utf-8",
    index=False,
)
