import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_segmentation_errors(image, true_mask, pred_mask, threshold=0.5, figsize=(15, 10)):
    """
    Visualize segmentation errors with color-coded overlays:
    - True Positives (TP): Green - correctly identified buildings
    - True Negatives (TN): Black/transparent - correctly identified non-buildings
    - False Positives (FP): Red - non-buildings incorrectly identified as buildings
    - False Negatives (FN): Blue - buildings incorrectly identified as non-buildings
    
    Args:
        image: Input image tensor [C, H, W] or [H, W, C]
        true_mask: Ground truth mask tensor [H, W]
        pred_mask: Predicted mask tensor [H, W] or probability map to be thresholded
        threshold: Threshold for converting probabilities to binary predictions
        figsize: Figure size for the plot
    """
    # Handle tensors and convert to numpy
    if torch.is_tensor(pred_mask):
        pred_mask = pred_mask.detach().cpu().numpy()
    if torch.is_tensor(true_mask):
        true_mask = true_mask.detach().cpu().numpy()
    if torch.is_tensor(image):
        if image.dim() == 3 and image.size(0) == 3:  # [C, H, W] format
            image = image.permute(1, 2, 0).detach().cpu().numpy()
        else:
            image = image.detach().cpu().numpy()
    
    # Convert probability map to binary mask if needed
    if pred_mask.max() > 1.0:
        pred_mask = (torch.sigmoid(torch.tensor(pred_mask)) > threshold).numpy()
    elif np.max(pred_mask) <= 1.0 and not np.all(np.isin(pred_mask, [0, 1])):
        pred_mask = (pred_mask > threshold).astype(np.float32)
    
    # Ensure true_mask is binary
    if not np.all(np.isin(true_mask, [0, 1])):
        true_mask = (true_mask > 0.5).astype(np.float32)
    
    # Compute error categories
    true_positive = (pred_mask == 1) & (true_mask == 1)
    true_negative = (pred_mask == 0) & (true_mask == 0)
    false_positive = (pred_mask == 1) & (true_mask == 0)
    false_negative = (pred_mask == 0) & (true_mask == 1)
    
    # Create color-coded error visualization
    error_vis = np.zeros((*true_mask.shape, 3), dtype=np.float32)
    error_vis[true_positive] = [0, 1, 0]    # Green for TP
    error_vis[false_positive] = [1, 0, 0]   # Red for FP
    error_vis[false_negative] = [0, 0, 1]   # Blue for FN
    # TN is not highlighted (remains black)
    
    # Create blended visualization with original image
    alpha = 0.5  # Transparency for the overlay
    blended = image.copy()
    mask = np.any(error_vis > 0, axis=2)
    blended[mask] = alpha * error_vis[mask] + (1 - alpha) * image[mask]
    
    # Calculate error metrics for this specific image
    num_tp = np.sum(true_positive)
    num_tn = np.sum(true_negative)
    num_fp = np.sum(false_positive)
    num_fn = np.sum(false_negative)
    
    precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0
    recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    iou = num_tp / (num_tp + num_fp + num_fn) if (num_tp + num_fp + num_fn) > 0 else 0
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    # Plot ground truth mask
    axes[0, 1].imshow(true_mask, cmap='gray')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Plot predicted mask
    axes[1, 0].imshow(pred_mask, cmap='gray')
    axes[1, 0].set_title(f'Prediction (threshold={threshold})')
    axes[1, 0].axis('off')
    
    # Plot error overlay on original image
    axes[1, 1].imshow(blended)
    axes[1, 1].set_title('Error Analysis')
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='green', label=f'True Positive: {num_tp}'),
        plt.Rectangle((0, 0), 1, 1, color='black', label=f'True Negative: {num_tn}'),
        plt.Rectangle((0, 0), 1, 1, color='red', label=f'False Positive: {num_fp}'),
        plt.Rectangle((0, 0), 1, 1, color='blue', label=f'False Negative: {num_fn}')
    ]
    axes[1, 1].legend(handles=legend_elements, loc='upper right', fontsize='small')
    axes[1, 1].axis('off')
    
    # Add metrics as text
    plt.figtext(0.5, 0.01, 
                f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | IoU: {iou:.4f}',
                ha='center', fontsize=12)
    
    plt.show()

    return {
        'tp': num_tp,
        'tn': num_tn,
        'fp': num_fp,
        'fn': num_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }