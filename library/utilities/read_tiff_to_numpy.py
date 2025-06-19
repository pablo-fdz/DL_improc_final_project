import torch
from torchvision import io
import numpy as np
from PIL import Image
import tifffile


def read_tiff_to_numpy(file_path: str, dtype=np.float32, normalize: bool = False) -> np.ndarray:
    """
    Imports tiff images as numpy arrays.
    """
    img = tifffile.imread(str(file_path))  # img is a np.array

    # If grayscale, add channel dimension
    if img.ndim == 2:
        img = img[None, :, :]  # [1, H, W]
    elif img.ndim == 3:
        img = img.transpose(2, 0, 1)  # [C, H, W]

    if normalize:
        img = img.astype(np.float32) / 255.0

    return img.astype(dtype)