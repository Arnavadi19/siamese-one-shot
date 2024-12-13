import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class OmniglotTrain(Dataset):

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.image_folder = []
        self._prepare_data()

    def _prepare_data(self):
        for root, dirs, files in os.walk(self.data_path):
            if files:
                images = [os.path.join(root, file) for file in files if file.endswith('.png')]
                if len(images) > 1:
                    self.image_folder.append(images)

    def __getitem__(self, index):
        folder = self.image_folder[index]
        img1_path, img2_path = torch.utils.data.random_split(folder, [1, len(folder)-1])
        img1 = Image.open(img1_path[0]).convert('L')
        img2 = Image.open(img2_path[0]).convert('L')
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(1 if img1_path[0].split('_')[1] == img2_path[0].split('_')[1] else -1)
        return img1, img2, label

    def __len__(self):
        return len(self.image_folder)

class OmniglotTest(Dataset):

    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.test_images = []
        self._prepare_data()

    def _prepare_data(self):
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if file.endswith('.png'):
                    self.test_images.append(os.path.join(root, file))

    def __getitem__(self, index):
        img_path = self.test_images[index]
        img = Image.open(img_path).convert('L')

        if self.transform:
            img = self.transform(img)

        return img, img_path

    def __len__(self):
        return len(self.test_images)

