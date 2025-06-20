from matplotlib import pyplot as plt
import torch

def display_image(image: torch.Tensor, title: str = '', cmap: str = 'gray', figsize=(4, 4), rgb_bands: tuple = None) -> None:
    # Convert to float32 if it's float16 (matplotlib doesn't support float16)
    if image.dtype == torch.float16:
        image = image.to(torch.float32)
    
    # If rgb_bands is specified and image has multiple channels, extract RGB bands
    if rgb_bands is not None and image.dim() == 3 and image.size(0) > 3:
        # Extract the specified RGB bands (assuming bands are in first dimension)
        image = image[list(rgb_bands), :, :]
        cmap = None  # Don't use colormap for RGB images
    
        # # Apply contrast stretching to enhance visibility (old version)
        # # Stretch each band to use full 0-1 range
        # for i in range(image.size(0)):
        #     band = image[i, :, :]
        #     min_val = torch.min(band)
        #     max_val = torch.max(band)
        #     if max_val > min_val:  # Avoid division by zero
        #         image[i, :, :] = (band - min_val) / (max_val - min_val)

        # Apply robust contrast stretching to enhance visibility
        # Stretch each band to use full 0-1 range after clipping outliers
        for i in range(image.size(0)):
            band = image[i, :, :]
            # Use percentiles to clip outliers, making the stretch more robust
            min_val = torch.quantile(band, 0.02)
            max_val = torch.quantile(band, 0.98)
            if max_val > min_val:  # Avoid division by zero
                # Clip the band to the percentile range
                band = torch.clamp(band, min_val, max_val)
                # Scale the clipped band to the [0, 1] range
                image[i, :, :] = (band - min_val) / (max_val - min_val)
            else:
                # Handle cases where the band is constant
                image[i, :, :] = torch.zeros_like(band)

    # Rearrange dimensions from (channels, height, width) to (height, width, channels)
    if image.dim() == 3:
        image = torch.einsum('dhw -> hwd', image)
        # Ensure RGB values are in [0, 1] range
        image = torch.clamp(image, 0, 1)
    
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image, cmap=cmap)
    ax.set_title(title, fontsize=15)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()
    plt.close()