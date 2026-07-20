import numpy as np
import torch
from torch.utils.data import Dataset


class StoryDataset(Dataset):

    def __init__(self, path, context_length, stride=64):

        self.data = np.memmap(
            path,
            dtype=np.uint32,
            mode="r"
        )

        self.context_length = context_length
        self.stride = stride

    def __len__(self):

        return (len(self.data) - self.context_length - 1) // self.stride

    def __getitem__(self, idx):

        idx *= self.stride

        x = torch.tensor(
            self.data[idx:idx+self.context_length],
            dtype=torch.long
        )

        y = torch.tensor(
            self.data[idx+1:idx+self.context_length+1],
            dtype=torch.long
        )

        return x, y