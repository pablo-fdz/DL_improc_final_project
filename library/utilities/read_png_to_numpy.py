import numpy as np
from PIL import Image


def read_png_to_numpy(file_path: str, dtype=np.float32, normalize: bool = False) -> np.ndarray:
    """
    Imports PNG images as numpy arrays.
    
    Args:
        file_path: Path to the .png file
        dtype: Numpy data type for the output array
        normalize: Whether to normalize pixel values to [0, 1] range
        
    Returns:
        numpy.ndarray: Image array with shape [C, H, W] for multi-channel or [1, H, W] for grayscale
    """
    # Use PIL to read PNG files
    img = Image.open(file_path)
    img = np.array(img)  # Convert PIL image to numpy array

    # If grayscale, add channel dimension
    if img.ndim == 2:
        img = img[None, :, :]  # [1, H, W]
    elif img.ndim == 3:
        img = img.transpose(2, 0, 1)  # [C, H, W]

    if normalize:
        img = img.astype(np.float32) / 255.0

    return img.astype(dtype)