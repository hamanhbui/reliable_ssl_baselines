import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import os
import pandas as pd
ds_test = tfds.load(
    name = 'cifar10_corrupted/impulse_noise_5',
    split= 'test'
)
numpy_items = tfds.as_numpy(ds_test)

test_paths = []
test_labels = []

for index, item in enumerate(numpy_items):
    directory = "/home/ubuntu/datasets/CIFAR-10/corruptions/impulse_noise_5/" + str(item['label'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = str(item['label']) + "/te_image_" + str(index) + ".png"
    cv2.imwrite("/home/ubuntu/datasets/CIFAR-10/corruptions/impulse_noise_5/" + path, item['image'])
    test_paths.append(path)
    test_labels.append(item['label'])

test_meta_files = pd.DataFrame({"path": test_paths, "label": test_labels})
test_meta_files.to_csv(
    "/home/ubuntu/datasets/CIFAR-10/metadata/impulse_noise_5_kfold.txt",
    header=None,
    sep=" ",
    encoding="utf-8",
    index=False,
)