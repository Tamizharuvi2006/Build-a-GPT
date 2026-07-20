import os
import glob

from config.model_config import CONTEXT_LENGTH
from config.train_config import *

from dataset.dataloader import create_dataloader

from model.llm import FantasyLLM

from training.optimizer import create_optimizer
from training.scheduler import create_scheduler
from training.trainer import Trainer
from training.checkpoint import load_checkpoint

from utils.seed import set_seed
from utils.device import get_device


def main():

    # Seed
    set_seed(SEED)

    # Device
    device = get_device()

    print(f"Using device: {device}")

    # Data
    train_loader = create_dataloader(
        "data/processed/train.bin",
        CONTEXT_LENGTH,
        BATCH_SIZE,
        shuffle=True
    )

    val_loader = create_dataloader(
        "data/processed/val.bin",
        CONTEXT_LENGTH,
        BATCH_SIZE,
        shuffle=False
    )

    print("Dataset loaded.\n")

    # Dataset Information
    print("========== DATASET INFO ==========")
    print(f"Training Samples   : {len(train_loader.dataset):,}")
    print(f"Validation Samples : {len(val_loader.dataset):,}")
    print(f"Training Batches   : {len(train_loader):,}")
    print(f"Validation Batches : {len(val_loader):,}")
    print(f"Batch Size         : {train_loader.batch_size}")
    print("==================================\n")

    # Model
    model = FantasyLLM()

    print("Model created.")

    # Optimizer
    optimizer = create_optimizer(
        model,
        LEARNING_RATE,
        WEIGHT_DECAY
    )

    # Scheduler
    scheduler = create_scheduler(
        optimizer,
        EPOCHS
    )

    # -----------------------------
    # Resume from latest checkpoint
    # -----------------------------
    start_epoch = 0

    checkpoint_files = glob.glob(
        os.path.join(CHECKPOINT_DIR, "epoch_*.pt")
    )

    if checkpoint_files:

        latest_checkpoint = max(
            checkpoint_files,
            key=os.path.getctime
        )

        start_epoch, last_loss = load_checkpoint(
            model,
            optimizer,
            latest_checkpoint,
            device
        )

        print(f"Resuming from Epoch {start_epoch}")
        print(f"Previous Validation Loss: {last_loss:.4f}\n")

    else:

        print("No checkpoint found. Starting fresh.\n")

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=CHECKPOINT_DIR
    )

    print("Starting training...\n")

    trainer.fit(
        EPOCHS,
        start_epoch
    )


if __name__ == "__main__":
    main()