from torch.utils.data import DataLoader
from dataset.dataset import StoryDataset


def create_dataloader(
    path,
    context_length,
    batch_size,
    shuffle=True,
    stride=64
):

    dataset = StoryDataset(
        path,
        context_length,
        stride
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True
    )