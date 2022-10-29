import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import os
import pandas as pd
import imutils

ds = tfds.load(
    name = 'mnist_corrupted/translate',
    split= 'test'
)
numpy_items = tfds.as_numpy(ds)

paths = []
labels = []

for index, item in enumerate(numpy_items):
    directory = "/home/ubuntu/datasets/MNIST/ood/translate/" + str(item['label'])
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = str(item['label']) + "/te_image_" + str(index) + ".png"
    # item['image'] = imutils.rotate(item['image'], angle=75)
    cv2.imwrite("/home/ubuntu/datasets/MNIST/ood/translate/" + path, item['image'])
    paths.append(path)
    labels.append(item['label'])

# meta_files = pd.DataFrame({"path": paths, "label": labels})
# meta_files.to_csv(
#     "/home/ubuntu/datasets/MNIST/metadata/motion_blur_kfold.txt",
#     header=None,
#     sep=" ",
#     encoding="utf-8",
#     index=False,
# )