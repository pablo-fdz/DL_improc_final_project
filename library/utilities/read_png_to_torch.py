import torch
from PIL import Image
import numpy as np


def read_png_to_torch(file_path: str, dtype: torch.dtype = torch.float32, normalize: bool = False) -> torch.Tensor:
    """
    Imports PNG images as torch tensors.
    
    Args:
        file_path: Path to the .png file
        dtype: PyTorch data type for the output tensor
        normalize: Whether to normalize pixel values to [0, 1] range
        
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
        img = img.astype(np.float32) / 255.0

    return torch.from_numpy(img).type(dtype)