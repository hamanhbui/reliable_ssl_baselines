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
        self.trans_hyp = [0, 180, 0.7, 1.3, 0.3, -0.3, 0, 2, -2]

    def get_image(self, sample_path):
        img = Image.open(sample_path)
        return self.image_transformer(img)

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        sample = self.get_image(self.path + self.sample_paths[index])
        ptrainy = np.zeros((10))
        rot_index = np.random.randint(2)
        scale_index = np.random.randint(2, 4)
        shear_index = np.random.randint(4, 6)
        transh_index = np.random.randint(6, 9)
        transw_index = np.random.randint(6, 9)

        # indices for translation
        trans_indices = []
        if transh_index == 7:
            trans_indices.append(6)
        elif transh_index == 8:
            trans_indices.append(7)

        if transw_index == 7:
            trans_indices.append(8)
        elif transw_index == 8:
            trans_indices.append(9)

        if 0.9 > random():
            pass
        else:
            sample = transforms.functional.affine(
                sample,
                angle=self.trans_hyp[rot_index],
                scale=self.trans_hyp[scale_index],
                translate=(self.trans_hyp[transw_index], self.trans_hyp[transh_index]),
                shear=self.trans_hyp[shear_index],
            )
            ptrainy[rot_index] = 1
            ptrainy[scale_index] = 1
            ptrainy[shear_index] = 1
            ptrainy[trans_indices] = 1

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
