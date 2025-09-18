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