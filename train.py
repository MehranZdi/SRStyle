import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from discriminator_model import Discriminator
from generator_model import Generator
from dataset import ImageDataset, get_dataloader
from tqdm import tqdm


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G_A = Generator(img_channels=3).to(device)
G_B = Generator(img_channels=3).to(device)
D_A = Discriminator(img_channel=3).to(device)
D_B = Discriminator(img_channel=3).to(device)

criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

optimizer_G = optim.Adam(G_A.parameters() + list(G_B.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

# lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 0.95 ** epoch)
# lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lambda epoch: 0.95 ** epoch)
# lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lambda epoch: 0.95 ** epoch)

writer = SummaryWriter()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

hr_dir = './data/HR_styled'
lr_dirs = ['./data/LR_x4', './data/LR_x3']

train_dataset = ImageDataset(hr_dir, lr_dirs)
train_loader = DataLoader(train_dataset, hr_dir, lr_dirs, batch_size=4, shuffle=True)

# Training
EPOCHS = 100
for epoch in range(EPOCHS):
    batch_iterator = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False)
