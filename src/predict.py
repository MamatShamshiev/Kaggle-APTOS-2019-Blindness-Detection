import imageio
import os

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.models import resnet34

from albumentations import (
    Compose, Resize, Normalize
)
from albumentations.torch import ToTensor

def pre_transforms(image_size=224):
    return Compose(
        [
            Resize(image_size, image_size),
        ]
    )

def post_transforms():
    return Compose([Normalize(), ToTensor()])


class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file, path_to_data, transform):
        self.data = pd.read_csv(csv_file)
        self.path_to_data = path_to_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path_to_data, self.data.loc[idx, 'id_code'])
        image = imageio.imread(img_name, pilmode="RGB")
        image = self.transform(image=image)
        return image

def predict(test_dl):
    with torch.no_grad():
        test_preds = np.zeros((len(test_dataset), 1))
        t = tqdm(test_dl)
        for i, x_batch in enumerate(t):
            pred = model(x_batch['image'].to(device))
            test_preds[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
    return test_preds

def round_preds(preds):
    coef = [0.5, 1.5, 2.5, 3.5]

    for i, pred in enumerate(preds):
        if pred < coef[0]:
            preds[i] = 0
        elif pred >= coef[0] and pred < coef[1]:
            preds[i] = 1
        elif pred >= coef[1] and pred < coef[2]:
            preds[i] = 2
        elif pred >= coef[2] and pred < coef[3]:
            preds[i] = 3
        else:
            preds[i] = 4
    return preds

path_to_data = "./data/test_images"
path_to_csv = "./data/test.csv"
path_to_checkpoint = "./logs/test/checkpoints/best.pth"
path_to_submission = "./data/sample_submission.csv"
image_size = 256
batch_size = 32
num_workers = 4

device = torch.device("cuda:0")
checkpoint = torch.load(path_to_checkpoint)
model = resnet34(num_classes=1)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

test_transforms = Compose([pre_transforms(image_size), post_transforms()])
test_dataset = RetinopathyDatasetTest(csv_file=path_to_csv, path_to_data=path_to_data, transform=test_transforms)

test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_preds = predict(test_dl)
test_preds = round_preds(test_preds)

sample = pd.read_csv(path_to_submission)
sample.diagnosis = test_preds.astype(int)
print(sample.head())
sample.to_csv("submission.csv", index=False)
