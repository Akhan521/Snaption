'''
Trainer module & utilities for Snaption image captioning model.
'''

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
from typing import Dict, Tuple, List
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

import snaption
from .dataset import ImageCaptioningDataset

class SnaptionTrainer:
    '''
    Trainer class for the Snaption image captioning model.
    '''
    def __init__(
        self,
        model: snaption.ImageCaptioner,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        vocab_mapper: snaption.VocabMapper | None = None,
        device: torch.device | str = 'cpu',
    ):
        '''
        Initialize the trainer with the appropriate components.

        Args:
            model (snaption.ImageCaptioner): The image captioning model to be trained.
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader | None): DataLoader for the validation dataset. Defaults to None.
            vocab_mapper (snaption.VocabMapper | None): Vocabulary mapper for decoding captions. Defaults to None.
            device (torch.device | str): Device to run the training on. Defaults to 'cpu'.
        '''
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocab_mapper = vocab_mapper
        self.device = device

        # Training state:
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # History tracking:
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.learning_rates: List[float] = []

        print(f"Trainer initialized on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

    def setup_training(
        self, 
        learning_rate: float = 2e-4,
        weight_decay: float = 5e-5,
        betas: Tuple[float, float] = (0.9, 0.98),
        label_smoothing: float = 0.1,
        max_epochs: int = 200,
        freeze_encoder: bool = True,
        use_scheduler: bool = True,
        scheduler_max_lr: float = 2e-4,
        scheduler_epochs: int = 200,
        scheduler_pct_start: float = 0.1,
    ):
        '''
        Setup the training configuration, including optimizer, scheduler, and loss function.

        Args:
            learning_rate (float): Learning rate for the optimizer. Defaults to 2e-4.
            weight_decay (float): Weight decay for the optimizer. Defaults to 5e-5.
            betas (Tuple[float, float]): Betas for the AdamW optimizer. Defaults to (0.9, 0.98).
            label_smoothing (float): Label smoothing factor for the loss function. Defaults to 0.1.
            max_epochs (int): Maximum number of training epochs. Defaults to 200.
            freeze_encoder (bool): Whether to freeze the image encoder during training. Defaults to True.
            use_scheduler (bool): Whether to use a learning rate scheduler. Defaults to True.
            scheduler_max_lr (float): Maximum learning rate for the scheduler. Defaults to 2e-4.
            scheduler_epochs (int): Total number of epochs for the scheduler. Defaults to 200.
            scheduler_pct_start (float): The percentage of the scheduler_epochs spent increasing the learning rate. Defaults to 0.1.
        '''
        if freeze_encoder:
            self.model.freeze_encoder()
            print("Image encoder frozen for training.")

        # Set up the loss function with label smoothing.
        self.criterion = nn.CrossEntropyLoss(
            ignore_index = self.vocab_mapper['<PAD>'],
            label_smoothing = label_smoothing
        )

        # Set up the optimizer.
        self.optimizer = AdamW(
            self.model.parameters(),
            lr = learning_rate,
            weight_decay = weight_decay,
            betas = betas
        )

        # Set up the learning rate scheduler if specified.
        if use_scheduler:
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr = scheduler_max_lr,
                steps_per_epoch = len(self.train_loader),
                epochs = scheduler_epochs,
                pct_start = scheduler_pct_start,
                anneal_strategy = 'cos'
            )
        else:
            self.scheduler = None

        self.max_epochs = max_epochs

        print("Training setup complete.")
        print(f"   - Learning Rate: {learning_rate}")
        print(f"   - Weight Decay: {weight_decay}")
        print(f"   - Max Epochs: {max_epochs}")
        print(f"   - Label Smoothing: {label_smoothing}")
        print(f"   - Scheduler: {'OneCycleLR' if use_scheduler else 'None'}")

    def train_epoch(self) -> float:
        '''Train the model for one epoch.'''
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        progress_bar = tqdm(
            self.train_loader,
            desc = f"Epoch {self.current_epoch + 1}/{self.max_epochs}",
            leave = False
        )

        for batch_idx, (images, captions) in enumerate(progress_bar):
            images = images.to(self.device)
            captions = captions.to(self.device)

            # Prepare autoregressive inputs and targets.
            inputs = captions[:, :-1] # All tokens except the last.
            targets = captions[:, 1:] # All tokens except the first.

            # Forward pass.
            outputs = self.model(images, inputs)

            # Compute loss.
            B, T, V = outputs.shape
            loss = self.criterion(
                outputs.reshape(B * T, V),
                targets.reshape(B * T)
            )

            # Backward pass.
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping.
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)

            # Optimizer step and scheduler step (if applicable).
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.learning_rates.append(current_lr)

            # Update metrics.
            total_loss += loss.item()
            self.global_step += 1

            # Update progress bar.
            if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'lr': f'{current_lr:.6f}' if self.scheduler else f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })

        avg_epoch_loss = total_loss / num_batches
        self.train_losses.append(avg_epoch_loss)
        self.current_epoch += 1

        return avg_epoch_loss
    
    def validate(self) -> float:
        '''Validate the model on the validation dataset.'''
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for images, captions in tqdm(self.val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                captions = captions.to(self.device)

                # Prepare autoregressive inputs and targets.
                inputs = captions[:, :-1] # All tokens except the last.
                targets = captions[:, 1:] # All tokens except the first.

                # Forward pass.
                outputs = self.model(images, inputs)

                # Compute loss.
                B, T, V = outputs.shape
                loss = self.criterion(
                    outputs.reshape(B * T, V),
                    targets.reshape(B * T)
                )

                total_loss += loss.item()

        avg_val_loss = total_loss / num_batches
        self.val_losses.append(avg_val_loss)

        return avg_val_loss

