import torch
from torchvision import io
import numpy as np
from PIL import Image
import tifffile


def read_tiff_to_torch(file_path: str, dtype: torch.dtype = torch.float32, normalize: bool = False) -> torch.Tensor:
    """
    Imports tiff images as a torch tensors.
    """
    img = tifffile.imread(str(file_path)) # img is a np.array

    # If grayscale, add channel dimension
    if img.ndim == 2:
        img = img[None, :, :]  # [1, H, W]
    elif img.ndim == 3:
        img = img.transpose(2, 0, 1)  # [C, H, W]

    if normalize:
        img = img.astype(np.float32) / 255.0

    return torch.from_numpy(img).type(dtype)