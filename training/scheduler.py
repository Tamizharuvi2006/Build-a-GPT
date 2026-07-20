from torch.optim.lr_scheduler import CosineAnnealingLR


def create_scheduler(
        optimizer,
        epochs
):

    return CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )