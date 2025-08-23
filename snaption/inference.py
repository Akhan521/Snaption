import torch
import cv2
import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path
from typing import List, Optional, Tuple

from .model import ImageCaptioner
from .utils import VocabMapper, WordTokenizer, cleanup_caption

class SnaptionModel:
    '''
    A high-level interface for generating image captions with Snaption.

    This class handles model loading, image preprocessing, and caption generation.
    '''
    def __init__(
        self,
        model_path: str,
        vocab_mapper: Optional[VocabMapper] = None,
        device: Optional[torch.device] = None,
    ):
        '''
        Initialize the SnaptionModel with the specified parameters.

        Args:
            model_path (str): The path to our pre-trained model.
            vocab_mapper (Optional[VocabMapper]): A vocabulary mapper for text preprocessing.
            device (Optional[torch.device]): The device to run the model on (CPU or GPU).
        '''
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.vocab_mapper = vocab_mapper
        self.transform = self._create_transform()
        
        # Model configuration.
        self.config = {
            'context_length': 20,
            'num_blocks': 6,
            'model_dim': 512,
            'num_heads': 16,
            'dropout_prob': 0.5
        }
        
        if model_path:
            self.load_model(model_path)

    def _create_transform(self) -> alb.Compose:
        pass

    def load_model(self, model_path: str):
        pass