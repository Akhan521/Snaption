import torch
import cv2
import numpy as np
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path
from typing import List, Tuple
import warnings

from .model import ImageCaptioner
from .utils import VocabMapper, WordTokenizer, cleanup_caption

class SnaptionModel:
    '''
    A high-level interface for generating image captions with Snaption.

    This class handles model loading, image preprocessing, and caption generation.
    '''
    def __init__(
        self,
        model_path: str | None = None,
        vocab_mapper: VocabMapper | None = None,
        device: torch.device | None = None,
    ):
        '''
        Initialize the SnaptionModel with the specified parameters.

        Args:
            model_path (str): The path to our pre-trained model.
            vocab_mapper (VocabMapper | None): A vocabulary mapper for text preprocessing.
            device (torch.device | None): The device to run the model on (CPU or GPU).
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

    def load_model(self, model_path: str, vocab_mapper: VocabMapper | None = None):
        '''
        Load the pre-trained model and vocabulary mapper.

        Args:
            model_path (str): The path to the pre-trained model.
            vocab_mapper (VocabMapper | None): A vocabulary mapper for text preprocessing.
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
    
    def caption(
        self,
        image_input: str | np.ndarray | Image.Image | Path,
        max_length: int | None = None,
        temperature: float = 1.0,
        clean_up: bool = True
    ) -> str:
        '''
        Generate a caption for a single image.

        Args:
            image_input (str | np.ndarray | Image.Image | Path): The input image to caption.
            max_length (int | None): The maximum length of the generated caption.
            temperature (float): The temperature to use for sampling.
            clean_up (bool): Whether to clean up the generated caption.

        Returns:
            str: The generated caption.
        '''
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Preprocess the image.
        image_tensor = self._preprocess_image(image_input).to(self.device)

        # Generate caption.
        with torch.no_grad():
            generated_ids = self.model.generate(
                image_tensor,
                self.vocab_mapper,
                max_length=max_length,
                temperature=temperature
            )

        # Decode to text.
        caption = self.vocab_mapper.decode(generated_ids[0]) # Remove batch dimension.

        # Clean up the caption (i.e. remove padding and special tokens).
        if clean_up:
            caption = cleanup_caption(caption)
            
        return caption
    
    def caption_batch(
        self,
        images: List[str | np.ndarray | Image.Image | Path],
        max_length: int | None = None,
        temperature: float = 1.0,
        clean_up: bool = True
    ) -> List[str]:
        '''
        Generate captions for a batch of images.

        Args:
            images (List[str | np.ndarray | Image.Image | Path]): The input images to caption.
            max_length (int | None): The maximum length of the generated captions.
            temperature (float): The temperature to use for sampling.
            clean_up (bool): Whether to clean up the generated captions.

        Returns:
            List[str]: The generated captions.
        '''
        if not self.model:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Process images in batches:
        batch_size = 4
        all_captions = []

        for i in range(0, len(images), batch_size):
            batch_imgs = images[i:i + batch_size]

            # Preprocess the batch.
            batch_tensors = []
            for img in batch_imgs:
                try:
                    tensor = self._preprocess_image(img)
                    batch_tensors.append(tensor)
                except Exception as e:
                    warnings.warn(f"Failed to process image {img}: {e}")
                    batch_tensors.append(None)

            # Filter out invalid tensors.
            batch_tensors = [t for t in batch_tensors if t is not None]
            if not batch_tensors:
                continue

            # Stack tensors into a batch.
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)

            # Generate captions for the batch.
            with torch.no_grad():
                generated_ids = self.model.generate(
                    batch_tensor,
                    self.vocab_mapper,
                    max_length=max_length,
                    temperature=temperature
                )

            # Decode to text.
            batch_captions = self.vocab_mapper.decode_batch(generated_ids)

            # Clean up the captions (i.e. remove padding and special tokens).
            if clean_up:
                batch_captions = [cleanup_caption(c) for c in batch_captions]

            all_captions.extend(batch_captions)

        return all_captions
    
# Convenience function for quick captioning.
def caption_image(
    image_path: str | Path,
    model_path: str,
    vocab_mapper: VocabMapper,
    **kwargs
) -> str:
    '''
    A convenience function for quick captioning of a single image.

    Args:
        image_path (str | Path): The input image to caption.
        model_path (str): The path to the model.
        vocab_mapper (VocabMapper): The vocabulary mapper.
        **kwargs: Additional keyword arguments to pass to the caption method.

    Returns:
        str: The generated caption.
    '''
    # Initialize the model.
    model = SnaptionModel(model_path, vocab_mapper)

    # Caption the image.
    return model.caption(image_path, **kwargs)
