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
        self.img_size = 224
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
        '''Create the image transformation/preprocessing pipeline.'''
        return alb.Compose([
            alb.Resize(self.img_size, self.img_size),
            alb.Normalize(),
            ToTensorV2()
        ])

    def load_model(self, model_path: str, vocab_mapper: Optional[VocabMapper] = None):
        '''
        Load the pre-trained model and vocabulary mapper.

        Args:
            model_path (str): The path to the pre-trained model.
            vocab_mapper (Optional[VocabMapper]): A vocabulary mapper for text preprocessing.
        '''
        if vocab_mapper:
            self.vocab_mapper = vocab_mapper

        if not self.vocab_mapper:
            raise ValueError("A vocab_mapper must be provided (either during initialization or via the load_model method).")
        
        # Init. the model.
        self.model = ImageCaptioner(
            context_length=self.config['context_length'],
            vocab_size=len(self.vocab_mapper),
            num_blocks=self.config['num_blocks'],
            model_dim=self.config['model_dim'],
            num_heads=self.config['num_heads'],
            dropout_prob=self.config['dropout_prob']
        )

        # Load pre-trained weights.
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f'Model loaded successfully from {model_path} on {self.device}.')

    def _preprocess_image(self, image_input: str | np.ndarray | Image.Image | Path) -> torch.Tensor:
        '''
        Preprocess the input image for model input.

        Args:
            image_input (str | np.ndarray | Image.Image | Path): The input image to preprocess.

        Returns:
            torch.Tensor: The preprocessed image tensor.
        '''
        # Handle different input types:
        if isinstance(image_input, (str, Path)):
            image = cv2.imread(str(image_input))
            if image is None:
                raise ValueError(f"Could not load image from path: {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            image = np.array(image_input)
            # Convert RGBA to RGB if necessary.
            if image.shape[-1] == 4: # RGBA
                image = image[:, :, :3] # Remove alpha channel.
        elif isinstance(image_input, np.ndarray):
            image = image_input.copy()
        else:
            raise TypeError("Unsupported image input type. Must be a file path, Numpy Array, or PIL Image.")

        # Apply transformations.
        transformed = self.transform(image=image)
        image = transformed['image']
        return image.unsqueeze(0) # Add batch dimension.