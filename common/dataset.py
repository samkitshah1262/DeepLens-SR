import numpy as np
import torch
from torch.utils.data import Dataset


class LensDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = np.load(self.file_paths[idx])  # Load .npy file
        img = torch.tensor(img, dtype=torch.float32)  # Convert to tensor
        # img = img.unsqueeze(1)  # Add channel dimension (1, H, W)

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        return img, label
