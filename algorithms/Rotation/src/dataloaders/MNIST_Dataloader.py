from random import random, sample

import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class MNISTDataloader(Dataset):
    def __init__(self, path, sample_paths, class_labels):
        self.image_transformer = transforms.Compose([transforms.Resize((32, 32))])
        self.tile_transformer = transforms.Compose([transforms.ToTensor()])
        self.path = path
        self.sample_paths, self.class_labels = sample_paths, class_labels
        self.trans_hyp = [0, 45, 90, 135, 180, 225, 270, 315]

    def get_image(self, sample_path):
        img = Image.open(sample_path)
        return self.image_transformer(img)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample = self.get_image(self.path + self.sample_paths[index])
        rot_index = np.random.randint(1, 8)

        if 0.9 > random():
            ptrainy = 0
        else:
            sample = transforms.functional.rotate(sample, angle=self.trans_hyp[rot_index])
            ptrainy = rot_index

        class_label = self.class_labels[index]
        return self.tile_transformer(sample), class_label, ptrainy


class MNIST_Test_Dataloader(MNISTDataloader):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self.image_transformer = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    def __getitem__(self, index):
        sample = self.get_image(self.path + self.sample_paths[index])
        class_label = self.class_labels[index]
        return sample, class_label
