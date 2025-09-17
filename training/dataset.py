'''
Training dataset for Snaption image captioning model.
Handles the Flickr8k dataset loading and preprocessing.
'''

import torch
import cv2
import pandas as pd
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from snaption.utils import WordTokenizer, VocabMapper
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

class ImageCaptioningDataset(Dataset):
    '''
    Dataset class for image captioning training.

    Supports the Flickr8k dataset format and provides data augmentation
    for training image captioning models.
    '''
    def __init__(
            self, 
            df: pd.DataFrame,
            images_dir: str | Path,
            split: str = 'train',
            img_size: int = 224,
            tokenizer: WordTokenizer | None = None,
            vocab_mapper: VocabMapper | None = None,
            context_length: int = 20
    ):
        '''
        Initializes the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing image filenames and captions.
            images_dir (str | Path): Directory where images are stored.
            split (str): Dataset split - 'train' or 'val'.
            img_size (int): Size to which images are resized.
            tokenizer: Tokenizer for processing captions.
            vocab_mapper: Vocabulary mapper for converting tokens to indices.
            context_length (int): Maximum length of caption sequences.
        '''
        self.df = df
        self.images_dir = Path(images_dir)
        self.split = split
        self.img_size = img_size
        self.tokenizer = tokenizer
        self.vocab_mapper = vocab_mapper
        self.context_length = context_length

        # Validate dataset components.
        self._validate_dataset()

        # Create the augmentation pipeline.
        self.transform = self._create_transform()

    def _create_transform(self) -> alb.Compose:
        '''
        Create the image transformation pipeline based on the dataset split.
        '''
        transforms = [alb.Resize(self.img_size, self.img_size)]

        if self.split == 'train':
            transforms.extend([
                alb.HorizontalFlip(p = 0.5),
                alb.RandomResizedCrop(
                    size = (self.img_size, self.img_size),
                    scale = (0.85, 1.0),
                    ratio = (0.9, 1.1), 
                    p = 0.5
                ),
                alb.ShiftScaleRotate(
                    shift_limit = 0.05, # Up to 5% translation shift
                    scale_limit = 0.1,  # Up to 10% scaling
                    rotate_limit = 15,  # Up to 15 degrees rotation
                    border_mode = cv2.BORDER_CONSTANT,
                    fill = 0, # Fill empty areas with black
                    p = 0.5
                ),
                alb.ColorJitter(
                    brightness = 0.2,
                    contrast = 0.2,
                    saturation = 0.2,
                    hue = 0.1,
                    p = 0.5
                ),
                alb.RandomBrightnessContrast(p = 0.3),
                alb.GaussianBlur(blur_limit = (3, 5), p = 0.2),
            ])

        # Final transformations for all splits:
        transforms.extend([
            alb.Normalize(),
            ToTensorV2()
        ])

        return alb.Compose(transforms)
    
    def _validate_dataset(self):
        '''Validate the dataset and its components.'''
        if self.tokenizer is None or self.vocab_mapper is None:
            raise ValueError("Tokenizer and vocab_mapper must be provided.")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory {self.images_dir} does not exist.")
        
        # Check for missing images:
        missing_imgs = []
        for img_name in self.df['image_filename']:
            img_path = self.images_dir / img_name
            if not img_path.exists():
                missing_imgs.append(img_name)

        if missing_imgs:
            warnings.warn(f"Found {len(missing_imgs)} missing images in the dataset.\n"
                          f"Missing images: {missing_imgs}")
            
    def __len__(self) -> int:
        '''Returns the total number of samples in the dataset.'''
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Retrieves the image and caption at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Transformed image tensor and caption tensor.
        '''
        img_filename, captions = self.df.iloc[idx]

        # Load and transform the image.
        img_path = self.images_dir / img_filename
        try:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                raise FileNotFoundError(f"Image file {img_path} not found or could not be opened.")
            image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            transformed_img = self.transform(image = image)['image']
        except Exception as e:
            warnings.warn(f"Error loading image {img_path}: {e}")
            # Return a black image tensor as a fallback.
            transformed_img = torch.zeros(3, self.img_size, self.img_size)

        # Process captions:
        encoded_captions = []
        for i, caption in enumerate(captions):
            tokens = self.tokenizer(caption)
            token_indices = self.vocab_mapper(tokens)

            # Add start and end tokens.
            token_indices = [self.vocab_mapper['<START>']] + token_indices + [self.vocab_mapper['<END>']]
            
            # Pad or truncate to context_length.
            if len(token_indices) < self.context_length:
                pad_length = self.context_length - len(token_indices)
                token_indices += [self.vocab_mapper['<PAD>']] * pad_length
            else:
                token_indices = token_indices[:self.context_length - 1] + [self.vocab_mapper['<END>']]

            encoded_captions.append(torch.tensor(token_indices, dtype = torch.long))

            # Randomly select one caption from the available captions.
            if encoded_captions:
                caption_idx = torch.randint(len(encoded_captions), size = (1,)).item()
                selected_caption = encoded_captions[caption_idx]
            else:
                # As a fallback, we'll return a placeholder caption.
                selected_caption = torch.tensor([
                    self.vocab_mapper['<START>'],
                    self.vocab_mapper['<UNKNOWN>'],
                    self.vocab_mapper['<END>']] + 
                    [self.vocab_mapper['<PAD>']] * (self.context_length - 3), 
                dtype=torch.long)

        return transformed_img, selected_caption