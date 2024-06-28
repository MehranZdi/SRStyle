import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, hr_dir, lr_dirs, transform=None):
        self.hr_images = list(os.path.join(hr_dir, file) for file in os.listdir(hr_dir))
        self.lr_images = list()
        for dir in lr_dirs:
            self.lr_images.extend([os.paht.join(dir, file) for file in os.listdir(dir)])

        self.transform = transform

    def __len__(self):
        return max(len(self.hr_images, self.lr_images))

    def __getitem__(self, idx):
        hr_image_path = random.choice(self.hr_images)
        lr_image_path = self.lr_images[idx % len(self.lr_images)]

        hr_image = Image.open(hr_image_path).convert('RGB')
        lr_image = Image.open(lr_image_path).convert('RGB')

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_dataloader(hr_dir, lr_dirs, batch_size, transform):
    dataset = ImageDataset(hr_dir, lr_dirs, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


hr_dir = 'home/mehran/SRStyle_Dataset/Styled'
lr_dirs = ['/home/mehran/SRStyle_Dataset/low_resx3', '/home/mehran/SRStyle_Dataset/low_resx4']

batch_size = 8
data_loader = get_dataloader(hr_dir, lr_dirs, batch_size, transform)
