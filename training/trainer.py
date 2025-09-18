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
        self.train_loss_history: List[float] = []
        self.val_loss_history: List[float] = []
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