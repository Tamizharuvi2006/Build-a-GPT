import os
import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    path
):
    # Create checkpoint folder if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)

    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    model,
    optimizer,
    path,
    device
):
    checkpoint = torch.load(
        path,
        map_location=device
    )

    target_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    target_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if optimizer is not None:
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    print(f"Loaded checkpoint: {path}")

    return (
        checkpoint["epoch"],
        checkpoint["loss"],
    )