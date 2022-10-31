import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import os
import pandas as pd
import imutils

def make_dataset(dataset_name):
    ds = tfds.load(
        name = 'cifar10_corrupted/' + dataset_name,
        split= 'test'
    )
    numpy_items = tfds.as_numpy(ds)

    paths = []
    labels = []

    for index, item in enumerate(numpy_items):
        directory = "/home/ubuntu/datasets/CIFAR-10/corruptions/" + dataset_name + "/" + str(item['label'])
        if not os.path.exists(directory):
            os.makedirs(directory)
        path = str(item['label']) + "/te_image_" + str(index) + ".png"
        # item['image'] = imutils.rotate(item['image'], angle=75)
        cv2.imwrite("/home/ubuntu/datasets/CIFAR-10/corruptions/" + dataset_name + "/" + path, item['image'])
        paths.append(path)
        labels.append(item['label'])

make_dataset("zoom_blur_5")
# meta_files = pd.DataFrame({"path": paths, "label": labels})
# meta_files.to_csv(
#     "/home/ubuntu/datasets/MNIST/metadata/test_kfold.txt",
#     header=None,
#     sep=" ",
#     encoding="utf-8",
#     index=False,
# )