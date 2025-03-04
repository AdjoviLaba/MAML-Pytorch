import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class EEGDataset(Dataset):
    def __init__(self, data_path, mode='train', resize=84):
        """
        Few-shot dataset loader for EEG spectrograms.

        :param data_path: Path to the dataset directory (e.g., /content/all_test_images/)
        :param mode: 'train', 'val', or 'test'
        :param resize: Image resize size
        """
        self.data = []
        self.labels = []
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((resize, resize)),  # Ensure images are 84x84
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
        ])

        # Load dataset from CSV file
        csv_file = os.path.join(data_path, 'test.csv') if mode != 'train' else os.path.join(data_path, 'train.csv')
        self.load_from_csv(data_path, csv_file)

    def load_from_csv(self, data_path, csv_file):
        """
        Loads dataset from a CSV file (used for MAML).
        """
        df = pd.read_csv(csv_file)
        label_map = {'hc': 0, 'sz': 1}  # Convert 'hc' to 0 and 'sz' to 1

        for _, row in df.iterrows():
            img_path = os.path.join(data_path, row['filename'])
            if os.path.exists(img_path):  # Check if file exists
                self.data.append(img_path)
                self.labels.append(label_map[row['label']])  # Convert label text to class index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        image = self.transform(image)  # Convert to grayscale tensor
        return image, torch.tensor(label, dtype=torch.long)
