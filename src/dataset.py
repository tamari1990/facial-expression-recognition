
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class FERDataset(Dataset):
    emotion_labels = {
        0: 'Angry',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happy',
        4: 'Sad',
        5: 'Surprise',
        6: 'Neutral'
    }

    def __init__(self, csv_path, transform=None):
        df = pd.read_csv(csv_path)
        self.emotions = df['emotion'].values
        self.pixels = df['pixels'].values
        self.transform = transform

    def __len__(self):
        return len(self.emotions)

    def __getitem__(self, idx):
        image = np.array(self.pixels[idx].split(), dtype=np.float32)
        image = image.reshape(48, 48)

        image = image / 255.0

        image = torch.tensor(image).unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.emotions[idx], dtype=torch.long)
        return image, label


def get_dataloaders(csv_path, batch_size=64, val_split=0.2):
    from torch.utils.data import random_split

    dataset = FERDataset(csv_path)

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
