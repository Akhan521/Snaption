import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional

class ImageCaptioner(nn.Module):
    '''
    Image captioning model using EfficientNet encoder and Transformer decoder.

    Args:
        context_length (int): Max sequence length for captions
        vocab_size (int): Size of the vocabulary
        num_blocks (int): Number of transformer blocks/decoder layers
        model_dim (int): Model dimension (embedding dimension)
        num_heads (int): Number of attention heads
        dropout_prob (float): Dropout probability
        encoder_model (str): Timm model name for encoder (default: efficientnet_b0)
    '''
    def __init__(
        self,
        context_length: int,
        vocab_size: int,
        num_blocks: int,
        model_dim: int,
        num_heads: int,
        dropout_prob: float,
        encoder_model: str = 'efficientnet_b0'
    ):
        super().__init__()

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.model_dim = model_dim

        # Timm CNN encoder:
        self.cnn_encoder = timm.create_model(encoder_model, pretrained=True)

        # Project CNN outputs to model_dim.
        temp_img = torch.zeros(1, 3, 224, 224) # To extract our CNN output's shape.
        with torch.no_grad():
            cnn_output = self.cnn_encoder(temp_img) # Shape: (B, in_features)
        in_features = cnn_output.shape[-1]
        self.project = nn.Linear(in_features, model_dim) # Project our cnn outputs of dim 'in_features' to 'model_dim'.

        # Embedding layers:
        self.word_embeddings = nn.Embedding(num_embeddings = vocab_size, embedding_dim = model_dim)
        self.pos_embeddings  = nn.Embedding(num_embeddings = context_length, embedding_dim = model_dim)

        # Defining our transformer decoder:
        decoder_layer = nn.TransformerDecoderLayer(
            d_model = model_dim,
            nhead = num_heads,
            dim_feedforward = 2 * model_dim,
            dropout = dropout_prob,
            batch_first = True, 
            norm_first = True # Apply layer norm. before self-attention and feedforward operations.
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer = decoder_layer,
            num_layers = num_blocks
        )

        # Vocabulary projection layer:
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        '''
        Forward pass for training.

        Args:
            images (torch.Tensor): Batch of images with shape (B, 3, H, W)
            captions (torch.Tensor): Batch of caption token IDs with shape (B, T)

        Returns:
            torch.Tensor: Output logits over vocabulary with shape (B, T, vocab_size)
        '''
        device = images.device
        B, T = captions.shape

        # Get embeddings.
        token_embeddings = self.word_embeddings(captions)
        positions = torch.arange(T, device=device)
        pos_embeddings = self.pos_embeddings(positions)
        total_embeddings = token_embeddings + pos_embeddings # Shape: (B, T, model_dim)

        # Encode our images.
        with torch.no_grad():
            cnn_output = self.cnn_encoder(images).view(B, -1) # Shape: (B, 3, H, W) -> (B, 3 * H * W)
            encoded_images = self.project(cnn_output) # Shape: (B, model_dim)

        # Our embedded captions have shape: (B, T, model_dim) while our encoded images have shape: (B, model_dim).
        # Our encoded images should have shape (B, 1, model_dim) to be compatible with the decoder.
        prepped_images_for_attn = torch.unsqueeze(encoded_images, dim = 1) 

        # Generate causal mask.
        attn_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device) # Shape: (T, T)

        # Apply transformer decoder.
        decoder_output = self.decoder(total_embeddings, prepped_images_for_attn, tgt_mask = attn_mask) # Shape: (B, T, model_dim)

        # Project decoder outputs to vocabulary space.
        vocab_proj_output = self.vocab_projection(decoder_output) # Shape: (B, T, vocab_size)

        return vocab_proj_output 
    
