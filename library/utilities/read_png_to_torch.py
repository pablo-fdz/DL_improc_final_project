import torch
from PIL import Image
import numpy as np


def read_png_to_torch(file_path: str, dtype: torch.dtype = torch.float32, normalize: bool = False, normalization_method='per_channel') -> torch.Tensor:
    """
    Imports PNG images as torch tensors.
    
    Args:
        file_path (str): Path to the .png file
        dtype (torch.dtype): PyTorch data type for the output tensor
        normalize (bool): Whether to normalize pixel values through a normalization method
        normalization_method (str): Method for normalization, options are '255', 'minmax', 'std', 'per_channel'.    
        
    Returns:
        torch.Tensor: Image tensor with shape [C, H, W] for multi-channel or [1, H, W] for grayscale
    """
    # Use PIL to read PNG files (better PNG support than tifffile)
    img = Image.open(file_path)
    img = np.array(img)  # Convert PIL image to numpy array

    # If grayscale, add channel dimension
    if img.ndim == 2:
        img = img[None, :, :]  # [1, H, W]
    elif img.ndim == 3:
        img = img.transpose(2, 0, 1)  # [C, H, W]

    if normalize:
        img = img.astype(np.float32)
        
        if normalization_method == '255':
            img = img / 255.0
        elif normalization_method == 'minmax':  # Normalize to [0, 1] range
            img_min, img_max = img.min(), img.max()
            if img_max - img_min == 0:
                # If all values are the same, set to 0 (or keep original values)
                img = np.zeros_like(img)
            else:
                img = (img - img_min) / (img_max - img_min)
        elif normalization_method == 'std':  # Normalize to zero mean and unit variance, same for all channels
            img_std = img.std()
            if img_std == 0:
                # If std is 0, just center the data (subtract mean)
                img = img - img.mean()
            else:
                img = (img - img.mean()) / img_std
        elif normalization_method == 'per_channel':  # Normalize each channel independently
            for c in range(img.shape[0]):
                channel = img[c]
                channel_mean = channel.mean()
                channel_std = channel.std()
                
                if channel_std == 0:
                    # If std is 0, just center the channel (subtract mean)
                    img[c] = channel - channel_mean
                else:
                    img[c] = (channel - channel_mean) / channel_std
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")

    return torch.from_numpy(img).type(dtype)