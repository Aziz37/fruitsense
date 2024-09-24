import os
import numpy as np
import torch
from torch.utils.data import Dataset

class FruitsDataset(Dataset):
    def __init__(self, data_folder, transform=None, split='train'):
        self.data_folder = data_folder
        if split == 'full':
            self.images = np.load(os.path.join(self.data_folder, 'X_full8.npy'))
            self.labels = np.load(os.path.join(self.data_folder, 'y_avsb.npy'))
        else:
            self.images = np.load(os.path.join(self.data_folder, 'X_' + split + '.npy'))
            self.labels = np.load(os.path.join(self.data_folder, 'y_' + split + '.npy'))

        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = np.asarray([self.labels[index]]).astype(np.float32)

        # Ihe image is in (H, W, C) format, so it needs to be converted to (C, H, W)
        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

class FruitsDatasetV2(Dataset):
    def __init__(self, data_folder, transform=None, split='train'):
        self.data_folder = data_folder
        self.images = np.load(os.path.join(self.data_folder, 'X_full8_' + split + '.npy'))
        self.labels = np.load(os.path.join(self.data_folder, 'y_avsb_' + split + '.npy'))

        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = np.asarray([self.labels[index]]).astype(np.float32)

        # Ihe image is in (H, W, C) format, so it needs to be converted to (C, H, W)
        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

class FruitsDatasetRGB(Dataset):
    def __init__(self, data_folder, transform=None, split='train'):
        self.data_folder = data_folder
        self.images = np.load(os.path.join(self.data_folder, 'X_rgb_' + split + '.npy'))
        self.labels = np.load(os.path.join(self.data_folder, 'y_avsb_' + split + '.npy'))

        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = np.asarray([self.labels[index]]).astype(np.float32)

        # Ihe image is in (H, W, C) format, so it needs to be converted to (C, H, W)
        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

class FruitsDataset30C(Dataset):
    def __init__(self, data_folder, transform=None, split='train'):
        self.data_folder = data_folder
        self.images = np.load(os.path.join(self.data_folder, 'X_full30_s_' + split + '.npy'))
        self.labels = np.load(os.path.join(self.data_folder, 'y_avsb_full30_s_' + split + '.npy'))

        self.transform = transform
        self.split = split

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        label = np.asarray([self.labels[index]]).astype(np.float32)

        # Ihe image is in (H, W, C) format, so it needs to be converted to (C, H, W)
        image = torch.tensor(image.transpose((2, 0, 1)), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label