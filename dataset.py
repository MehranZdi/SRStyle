import os
import random
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dirs, transform=None):
        self.hr_images = list(os.path.join(hr_dir, file) for file in os.listdir(hr_dir))
        self.lr_images = list()
        for dir in lr_dirs:
            self.lr_images.extend([os.path.join(dir, file) for file in os.listdir(dir)])

        self.transform = transform

    def __len__(self):
        return min(len(self.hr_images), len(self.lr_images))

    def __getitem__(self, idx):
        hr_image_path = random.choice(self.hr_images)
        lr_image_path = self.lr_images[idx % len(self.lr_images)]

        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image
