import os
from torchvision import transforms
import torch
from PIL import Image
import os

class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        if not os.path.exists(data_path):
            raise OSError(data_path + ' does not exist!')

        self.data = []

        folders = os.listdir(data_path)
        for folder in folders:

            full_path = os.path.join(data_path, folder)
            images = os.listdir(full_path)

            current_data = [os.path.join(full_path, image) for image in images]
            self.data += current_data

        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.transform_grayscale = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data[idx]
        image = Image.open(image_path).convert('RGB')
        try:
            image = self.transform(image)
        except RuntimeError:
            # Some grayscale images cause an error in normalize.
            image = self.transform_grayscale(image)
        return {'image': image, 'image_path': image_path}



def collate_skip_empty(batch):
    batch = [sample for sample in batch if sample] # check that sample is not None
    return torch.utils.data.dataloader.default_collate(batch)
