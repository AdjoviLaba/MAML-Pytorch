import os
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class EEGDataset(Dataset):
    def __init__(self, data_path, mode='train', n_way=2, k_shot=5, k_query=15, batchsz=50, resize=84, use_csv=False):
        """
        Few-shot dataset loader for EEG spectrograms.

        :param data_path: Path to the dataset directory (e.g., /content/all_test_images/)
        :param mode: 'train', 'val', or 'test'
        :param n_way: Number of classes per episode
        :param k_shot: Number of support examples per class
        :param k_query: Number of query examples per class
        :param batchsz: Number of tasks per batch
        :param resize: Image resize size
        :param use_csv: If True, uses test.csv for loading images (for MAML setup)
        """
        self.data = []
        self.labels = []
        self.resize = resize
        self.use_csv = use_csv
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((resize, resize)),  # Ensure images are 84x84
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
        ])

        # Define class label mapping
        self.label_map = {'hc': 0, 'sz': 1}  # Healthy Control (HC) = 0, Schizophrenia (SZ) = 1

        if use_csv:
            # Load dataset from CSV file
            csv_file = os.path.join(data_path, 'test.csv')
            self.load_from_csv(data_path, csv_file)
        else:
            # Load dataset from folder structure
            self.load_from_folders(data_path, mode)

    def load_from_folders(self, data_path, mode):
        """
        Loads data from structured folders (train/class_0/, train/class_1/ etc.).
        """
        class_dirs = [os.path.join(data_path, mode, f'class_{i}') for i in range(2)]  # class_0 (HC), class_1 (SZ)
        for class_idx, class_dir in enumerate(class_dirs):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.data.append(img_path)
                self.labels.append(class_idx)  # class_0 = 0, class_1 = 1

    def load_from_csv(self, data_path, csv_file):
        """
        Loads dataset from a CSV file (used for MAML).
        """
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            img_path = os.path.join(data_path, row['label'], row['filename'])
            if os.path.exists(img_path):  # Check if file exists
                self.data.append(img_path)
                self.labels.append(self.label_map[row['label']])  # Convert label text to class index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        label = self.labels[idx]
        image = Image.open(img_path)
        image = self.transform(image)  # Convert to grayscale tensor
        return image, torch.tensor(label, dtype=torch.long)
