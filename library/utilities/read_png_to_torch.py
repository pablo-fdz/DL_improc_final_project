import torch
from PIL import Image
import numpy as np


def read_png_to_torch(file_path: str, dtype: torch.dtype = torch.float32, normalize: bool = False, normalization_method='per_channel') -> torch.Tensor:
    """
    Imports PNG images as torch tensors. Handles NaNs and invalid values and 
    normalizes pixel values if specified.
    
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
            img_normalized = np.zeros_like(img, dtype=np.float32)
            valid_pixels = np.isfinite(img)
            img_normalized[valid_pixels] = img[valid_pixels] / 255.0
            img = img_normalized
        elif normalization_method == 'minmax':  # Normalize to [0, 1] range
            img_normalized = np.zeros_like(img, dtype=np.float32)
            valid_pixels = np.isfinite(img)

            if np.any(valid_pixels):
                img_valid = img[valid_pixels]
                img_min, img_max = img_valid.min(), img_valid.max()
                if (img_max - img_min) > 0:
                    img_normalized[valid_pixels] = (img_valid - img_min) / (img_max - img_min)
            img = img_normalized
        elif normalization_method == 'std':  # Normalize to zero mean and unit variance, same for all channels
            img_normalized = np.zeros_like(img, dtype=np.float32)
            valid_pixels = np.isfinite(img)

            if np.any(valid_pixels):
                img_valid = img[valid_pixels]
                img_mean = img_valid.mean()
                img_std = img_valid.std()
                if img_std > 0:
                    img_normalized[valid_pixels] = (img_valid - img_mean) / img_std
                else:
                    # If std is 0, just center the data
                    img_normalized[valid_pixels] = img_valid - img_mean
            img = img_normalized
        elif normalization_method == 'per_channel':  # Normalize each channel independently
            img_normalized = np.zeros_like(img, dtype=np.float32)
            for c in range(img.shape[0]):
                channel = img[c]
                valid_pixels = np.isfinite(channel)

                if not np.any(valid_pixels):
                    continue

                channel_valid = channel[valid_pixels]
                
                channel_mean = channel_valid.mean()
                channel_std = channel_valid.std()
                
                if channel_std > 0:
                    img_normalized[c, valid_pixels] = (channel_valid - channel_mean) / channel_std
                else:
                    # If std is 0, just center the data
                    img_normalized[c, valid_pixels] = channel_valid - channel_mean
            img = img_normalized
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")

    return torch.from_numpy(img).type(dtype)