import torch
import matplotlib.pyplot as plt

def visualize_predictions(images, true_masks, pred_probs, num_samples=3, threshold=0.5):
    """
    Visualize input images, true masks, and predicted masks.
    
    Args:
        images: Tensor of input images [N, C, H, W]
        true_masks: Tensor of true masks [N, H, W]
        pred_probs: Tensor of predicted probabilities [N, H, W]
        num_samples: Number of random samples to visualize
        threshold: Threshold for converting probabilities to binary masks
    """
    # Get random indices
    indices = torch.randperm(len(images))[:num_samples]
    
    # Create binary predictions from probabilities
    pred_masks = (pred_probs > threshold).float()
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    for i, idx in enumerate(indices):
        # Get the image, true mask, and predicted mask
        img = images[idx].permute(1, 2, 0)  # CHW -> HWC for matplotlib
        true_mask = true_masks[idx]
        pred_mask = pred_masks[idx]
        
        # Plot input image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Plot true mask
        axes[i, 1].imshow(true_mask, cmap='gray')
        axes[i, 1].set_title('True Mask')
        axes[i, 1].axis('off')
        
        # Plot predicted mask
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title(f'Predicted Mask (threshold={threshold})')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.show()