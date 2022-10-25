import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision

class CIFAR10Dataloader(Dataset):
    def __init__(self, path, sample_paths, class_labels):
        self.grid_size = 3
        self.permutations = self.retrieve_permutations(30)
        def make_grid(x):
            return torchvision.utils.make_grid(x, self.grid_size, padding=0)
        self.returnFunc = make_grid
        self.image_transformer = transforms.Compose(
            [transforms.Resize((33, 33))])
        self.tile_transformer = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        self.path = path
        self.sample_paths, self.class_labels = sample_paths, class_labels

    def get_image(self, sample_path):
        img = Image.open(sample_path)
        return self.image_transformer(img)

    def __len__(self):
        return len(self.sample_paths)

    def get_tile(self, img, n):
        w = float(img.size[0]) / self.grid_size
        y = int(n / self.grid_size)
        x = n % self.grid_size
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        tile = self.tile_transformer(tile)
        return tile

    def retrieve_permutations(self, classes):
        all_perm = np.load('/home/ubuntu/reliable_ssl_baselines/algorithms/Jigsaw/src/dataloaders/permutations_%d.npy' % (classes))
        return all_perm

    def __getitem__(self, index):
        sample = self.get_image(self.path + self.sample_paths[index])

        n_grids = self.grid_size ** 2
        tiles = [None] * n_grids
        for n in range(n_grids):
            tiles[n] = self.get_tile(sample, n)

        order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        if order == 0:
            data = tiles
        else:
            data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]

        data = torch.stack(data, 0)

        class_label = self.class_labels[index]
        return self.returnFunc(data), class_label, order


class CIFAR10_Test_Dataloader(CIFAR10Dataloader):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self.image_transformer = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    
    def __getitem__(self, index):
        sample = self.get_image(self.path + self.sample_paths[index])
        class_label = self.class_labels[index]
        return sample, class_label
