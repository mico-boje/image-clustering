from tqdm import tqdm
import cv2
import torch
import numpy as np
from src.data.twitter_image_dataset import TwitterDataset
from src.models.resnet import ResNet101

def get_features(image_path, batch):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # initialize our implementation of ResNet
    model = ResNet101(pretrained=True)
    model.eval()
    model.to(device)

    # read the dataset and initialize the data loader
    ds = TwitterDataset(image_path)
    dataloader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)

    features = None
    image_paths = []

    for batch in tqdm(dataloader, desc='Running the model inference'):
        images = batch['image'].to(device)
        image_paths += batch['image_path']

        with torch.no_grad():
            output = model.forward(images)

        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    return features, image_paths

