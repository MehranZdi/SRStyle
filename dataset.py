from PIL import Image
import os
import random
from torch.utils.data import Dataset
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dirs, transform, use_transform=True):
        self.hr_images = list(os.path.join(hr_dir, file) for file in os.listdir(hr_dir))
        self.lr_images = list()
        self.use_transform = use_transform
        for dir in lr_dirs:
            self.lr_images.extend([os.path.join(dir, file) for file in os.listdir(dir)])
        self.transform = transform


        self.length_dataset = max(len(self.hr_images), len(self.lr_images)) # 1000, 1500
        self.hr_len = len(self.hr_images)
        self.lr_len = len(self.lr_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        hr_image_path = random.choice(self.hr_images)
        lr_image_path = self.lr_images[index % self.lr_len]

        hr_image = np.array(Image.open(hr_image_path).convert("RGB"))
        lr_image = np.array(Image.open(lr_image_path).convert("RGB"))

        if self.use_transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return hr_image, lr_image
