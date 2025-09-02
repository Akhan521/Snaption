"""
Snaption: AI-Powered Image Captioning

A PyTorch-based image captioning package using an EfficientNet encoder 
and Transformer decoder architecture.

Example usage:
    >>> import snaption
    >>> model = snaption.SnaptionModel('path/to/weights.pt', vocab_mapper)
    >>> caption = model.caption('path/to/image.jpg')
    >>> print(caption)
"""

from .model import ImageCaptioner
from .inference import SnaptionModel, caption_image
from .utils import WordTokenizer, VocabMapper, build_vocab, prepare_caption, cleanup_caption

__version__ = "0.1.0"
__author__ = "Aamir Khan"
__email__ = "aamirksfg@gmail.com"

# What gets imported with "from snaption import *"
__all__ = [
    'ImageCaptioner',
    'SnaptionModel', 
    'caption_image',
    'WordTokenizer',
    'VocabMapper',
    'build_vocab',
    'prepare_caption',
    'cleanup_caption'
]