import torch
from torchvision import transforms


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
LAMBDA_CYCLE = 10
NUM_WORKERS = 4
NUM_EPOCHS = 5
LOAD_MODEL = False
SAVE_MODEL = True

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.RandomCrop(256),     # Crop randomly to 256x256
    transforms.ToTensor(),          # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                         std=[0.229, 0.224, 0.225]),
    ])