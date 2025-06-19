import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_predictions(images, true_masks, predictions, num_samples=3, num_classes=2, threshold=0.5, rgb_bands=(3, 2, 1)):
    """
    Visualize input images, true masks, and predicted masks for segmentation.
    Handles both binary and multi-class scenarios.
    
    Args:
        images (torch.Tensor): Tensor of input images [N, C, H, W].
        true_masks (torch.Tensor): Tensor of true masks [N, H, W].
        predictions (torch.Tensor): Tensor of model outputs.
                                   - For binary (num_classes=2), these are probabilities [N, H, W].
                                   - For multi-class, these are class indices [N, H, W].
        num_samples (int): Number of random samples to visualize.
        num_classes (int): Number of classes to determine visualization logic.
        threshold (float): Threshold for binary classification. Ignored for multi-class.
        rgb_bands (tuple): The indices of the R, G, B bands for displaying the input image.
    """
    # Get random indices
    if len(images) < num_samples:
        num_samples = len(images)
    indices = torch.randperm(len(images))[:num_samples]
    
    is_multiclass = num_classes > 2
    
    # Create binary predictions from probabilities if binary classification
    if not is_multiclass:
        pred_masks = (predictions > threshold).float()
    else:
        pred_masks = predictions # For multi-class, predictions are already class indices
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0) # Make it iterable if only one sample

    for i, idx in enumerate(indices):
        # --- 1. Prepare Input Image ---
        img = images[idx].cpu()
        
        # Extract RGB bands for visualization and apply contrast stretching
        if rgb_bands is not None and img.dim() == 3 and img.shape[0] >= max(rgb_bands)+1:
            img_display = img[list(rgb_bands), :, :].clone()
            for band_idx in range(img_display.size(0)):
                band = img_display[band_idx, :, :]
                min_val, max_val = torch.min(band), torch.max(band)
                if max_val > min_val:
                    img_display[band_idx, :, :] = (band - min_val) / (max_val - min_val)
            img_display = img_display.permute(1, 2, 0) # CHW -> HWC
        else:
            img_display = img.permute(1, 2, 0) 

        img_display = torch.clamp(img_display, 0, 1)

        # --- 2. Prepare Masks ---
        true_mask = true_masks[idx].cpu()
        pred_mask = pred_masks[idx].cpu()
        
        # --- 3. Plotting ---
        # Plot input image
        axes[i, 0].imshow(img_display)
        axes[i, 0].set_title('Input Image (RGB)')
        axes[i, 0].axis('off')
        
        # Plot true mask
        cmap = 'jet' if is_multiclass else 'gray'
        vmin = 0
        vmax = num_classes - 1 if is_multiclass else 1
        
        im1 = axes[i, 1].imshow(true_mask, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')
        
        # Plot predicted mask
        im2 = axes[i, 2].imshow(pred_mask, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

    # Add a colorbar for multi-class visualization
    if is_multiclass and num_samples > 0:
        fig.colorbar(im2, ax=axes[:, 2], orientation='vertical', fraction=0.05, pad=0.04, ticks=range(num_classes))

    plt.tight_layout()
    plt.show()
    plt.close(fig)