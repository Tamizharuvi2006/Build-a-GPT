import torch
from tqdm import tqdm

from training.loss import LanguageModelLoss
from training.metrics import perplexity
from training.checkpoint import save_checkpoint
from training.mixed_precision import MixedPrecisionManager
from training.gradient_accumulation import GradientAccumulator
from utils.logger import TrainingLogger
from training.callbacks.early_stopping import EarlyStopping
from training.callbacks.best_model import BestModelSaver
from training.callbacks.lr_monitor import LRMonitor
from training.callbacks.gradient_monitor import GradientMonitor

from config.train_config import (
    GRAD_CLIP,
    MAX_TRAIN_BATCHES,
    MAX_VAL_BATCHES,
    SAVE_EVERY,
)


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device,
        checkpoint_dir,
        callbacks=None,
        use_amp=True,
        accumulation_steps=4
    ):

        self.model = model.to(device)
        
        # Enable multi-GPU training if available
        if "cuda" in str(device) and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs for parallel training!")
            self.model = torch.nn.DataParallel(self.model)

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.criterion = LanguageModelLoss()

        self.checkpoint_dir = checkpoint_dir
        
        self.logger = TrainingLogger()
        self.amp = MixedPrecisionManager(enabled=use_amp)
        self.accumulator = GradientAccumulator(accumulation_steps=accumulation_steps)
        
        # Auto-Tune Memory/Batch Size based on VRAM
        if "cuda" in str(device):
            total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            print(f"[Auto-Tuner] Detected {total_vram_gb:.1f} GB of VRAM on primary GPU.")
            if total_vram_gb < 8.0:
                print(f"[Auto-Tuner] VRAM is low (< 8GB). Automatically doubling gradient accumulation steps from {self.accumulator.accumulation_steps} to {self.accumulator.accumulation_steps * 2} to prevent Out-of-Memory (OOM) errors.")
                self.accumulator.accumulation_steps *= 2
        
        # Instantiate default callbacks if none provided
        if callbacks is None:
            self.callbacks = [
                EarlyStopping(patience=5),
                BestModelSaver(checkpoint_dir=checkpoint_dir),
                LRMonitor(optimizer=self.optimizer),
                GradientMonitor(model=self.model)
            ]
        else:
            self.callbacks = callbacks

    def train_epoch(self):

        self.model.train()

        total_loss = 0
        batches = 0

        progress = tqdm(
            self.train_loader,
            total=MAX_TRAIN_BATCHES,
            desc="Training",
            leave=False
        )
        
        self.optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(progress):

            if batch_idx >= MAX_TRAIN_BATCHES:
                break

            x = x.to(self.device)
            y = y.to(self.device)

            with self.amp.context(self.device):
                out = self.model(x)
                if isinstance(out, tuple):
                    logits, moe_loss = out
                else:
                    logits = out
                    moe_loss = 0.0
                    
                loss = self.criterion(logits, y)
                loss = loss + 0.01 * moe_loss
                
                scaled_loss = self.accumulator.scale_loss(loss)

            self.amp.scale_loss(scaled_loss).backward()

            if self.accumulator.should_step(batch_idx):
                self.amp.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    GRAD_CLIP
                )
                self.amp.unscale_and_step(self.optimizer)
                self.optimizer.zero_grad()

            # For gradient monitor check
            for cb in self.callbacks:
                if isinstance(cb, GradientMonitor) and self.accumulator.should_step(batch_idx):
                    cb.check_for_anomalies()

            total_loss += loss.item()
            batches += 1

            progress.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        self.scheduler.step()

        return total_loss / batches

    def validate(self):

        self.model.eval()

        total_loss = 0
        batches = 0

        with torch.no_grad():

            for batch_idx, (x, y) in enumerate(self.val_loader):

                if batch_idx >= MAX_VAL_BATCHES:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)

                loss = self.criterion(logits, y)

                total_loss += loss.item()
                batches += 1

        return total_loss / batches

    def fit(self, epochs, start_epoch=0):

        end_epoch = start_epoch + epochs

        for epoch in range(start_epoch, end_epoch):

            train_loss = self.train_epoch()
            val_loss = self.validate()
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"\nEpoch {epoch + 1}/{end_epoch}")
            print(f"Train Loss      : {train_loss:.4f}")
            print(f"Validation Loss : {val_loss:.4f}")
            print(f"Perplexity      : {perplexity(val_loss):.2f}")
            
            # Log metrics using TrainingLogger
            self.logger.log_metrics(
                epoch=epoch + 1, 
                train_loss=train_loss, 
                val_loss=val_loss, 
                perplexity=perplexity(val_loss), 
                lr=current_lr
            )

            if (epoch + 1) % SAVE_EVERY == 0:
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    val_loss,
                    f"{self.checkpoint_dir}/epoch_{epoch+1}.pt"
                )
                
            # Process callbacks
            stop_training = False
            for cb in self.callbacks:
                if isinstance(cb, EarlyStopping):
                    if cb(val_loss):
                        stop_training = True
                elif isinstance(cb, BestModelSaver):
                    cb(self.model, self.optimizer, epoch + 1, val_loss)
                elif isinstance(cb, LRMonitor):
                    cb.log()

            if stop_training:
                print("Early stopping triggered. Halting training.")
                break

            print("-" * 50)