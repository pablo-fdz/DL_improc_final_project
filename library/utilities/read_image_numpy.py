import torch
from torchvision import io
import numpy as np

def read_image_numpy(file: str, dtype: torch.dtype, normalize: bool = True) -> np.ndarray:
    
    """
    Read a JPEG image file as a numpy array.
    
    Args:
        file (str): Path to the image file
        normalize (bool): Whether to normalize values to 0-1 range
        
    Returns:
        np.ndarray: Image array with shape [channels, height, width]
    """

    # Use torchvision's io.read_image and convert to numpy
    image = io.read_image(file).numpy()
    
    # For label images, round values to restore binary nature (for JPEG compression artifacts)
    if 'label' in file.lower():
        # Round to nearest of 0 or 255 - first find which value is closer
        image = np.where(image > 255/2, 255, 0)
    
    # Normalize the pixel values to 0-1 range if requested
    if normalize:
        image = image / 255.0
    
    image = torch.from_numpy(image).type(dtype)
    
    return image