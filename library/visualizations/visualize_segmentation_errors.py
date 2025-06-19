import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def visualize_segmentation_errors(image, true_mask, pred_mask, num_classes=2, threshold=0.5, figsize=(15, 10), rgb_bands=None):
    """
    Visualize segmentation errors for both binary and multi-class scenarios.
    - For binary cases, highlights misclassifications in Green (TP), Red (FP), and Blue (FN) visualization.
    - For multi-class cases, it will simply highlight all misclassified pixels in red.
    
    Args:
        image (torch.Tensor): Input image tensor [C, H, W].
        true_mask (torch.Tensor): Ground truth mask tensor [H, W].
        pred_mask (torch.Tensor): Predicted mask tensor.
                                  - For binary, probability map [H, W].
                                  - For multi-class, class indices [H, W].
        num_classes (int): Number of classes.
        threshold (float): Threshold for binary classification. Ignored for multi-class.
        figsize (tuple): Figure size.
        rgb_bands (tuple): Indices for R, G, B bands for multi-band images.
    """
    # --- 1. Data Preparation ---
    true_mask_np = true_mask.cpu().numpy() if torch.is_tensor(true_mask) else true_mask
    pred_mask_np = pred_mask.cpu().numpy() if torch.is_tensor(pred_mask) else pred_mask
    
    # Prepare image for display (handle multi-band and apply contrast stretch)
    img_display = image.cpu().clone()
    if rgb_bands is not None and img_display.dim() == 3 and img_display.shape[0] >= max(rgb_bands) + 1:
        img_display = img_display[list(rgb_bands), :, :]
        for band_idx in range(img_display.size(0)):
            band = img_display[band_idx, :, :]
            min_val, max_val = torch.min(band), torch.max(band)
            if max_val > min_val:
                img_display[band_idx, :, :] = (band - min_val) / (max_val - min_val)
    img_display = img_display.permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)

    is_multiclass = num_classes > 2
    
    # --- 2. Plotting Setup ---
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes[0, 0].imshow(img_display)
    axes[0, 0].set_title('Input Image (RGB)')
    axes[0, 0].axis('off')
    
    # --- 3. Scenario-specific Logic ---
    if is_multiclass:
        # --- Multi-class Error Analysis ---
        correct_pixels = (pred_mask_np == true_mask_np)
        error_pixels = ~correct_pixels
        
        error_vis = np.zeros((*true_mask_np.shape, 3), dtype=np.float32)
        error_vis[error_pixels] = [1, 0, 0]  # Red for misclassified pixels
        
        alpha = 0.5
        blended = img_display.copy()
        mask = np.any(error_vis > 0, axis=2)
        blended[mask] = alpha * error_vis[mask] + (1 - alpha) * img_display[mask]
        
        accuracy = np.sum(correct_pixels) / correct_pixels.size
        
        cmap, vmin, vmax = 'jet', 0, num_classes - 1
        axes[0, 1].imshow(true_mask_np, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, 1].set_title('Ground Truth Mask')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(pred_mask_np, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, 0].set_title('Predicted Mask')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(blended)
        axes[1, 1].set_title('Error Analysis')
        axes[1, 1].legend(handles=[Patch(facecolor='red', label='Misclassified Pixel')], loc='upper right')
        axes[1, 1].axis('off')
        
        plt.figtext(0.5, 0.01, f'Pixel Accuracy: {accuracy:.4f}', ha='center', fontsize=12)
        metrics = {'accuracy': accuracy}
    else:
        # --- Binary Error Analysis ---
        pred_mask_binary = (pred_mask_np > threshold).astype(np.float32) if not np.all(np.isin(pred_mask_np, [0, 1])) else pred_mask_np
        
        tp = (pred_mask_binary == 1) & (true_mask_np == 1)
        fp = (pred_mask_binary == 1) & (true_mask_np == 0)
        fn = (pred_mask_binary == 0) & (true_mask_np == 1)
        
        error_vis = np.zeros((*true_mask_np.shape, 3), dtype=np.float32)
        error_vis[tp], error_vis[fp], error_vis[fn] = [0, 1, 0], [1, 0, 0], [0, 0, 1]
        
        alpha = 0.5
        blended = img_display.copy()
        mask = np.any(error_vis > 0, axis=2)
        blended[mask] = alpha * error_vis[mask] + (1 - alpha) * img_display[mask]
        
        num_tp, num_fp, num_fn = np.sum(tp), np.sum(fp), np.sum(fn)
        precision = num_tp / (num_tp + num_fp) if (num_tp + num_fp) > 0 else 0
        recall = num_tp / (num_tp + num_fn) if (num_tp + num_fn) > 0 else 0
        iou = num_tp / (num_tp + num_fp + num_fn) if (num_tp + num_fp + num_fn) > 0 else 0
        
        axes[0, 1].imshow(true_mask_np, cmap='gray')
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(pred_mask_binary, cmap='gray')
        axes[1, 0].set_title(f'Prediction (threshold={threshold})')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(blended)
        axes[1, 1].set_title('Error Analysis')
        axes[1, 1].legend(handles=[Patch(facecolor='green', label='True Positive'), Patch(facecolor='red', label='False Positive'), Patch(facecolor='blue', label='False Negative')], loc='upper right')
        axes[1, 1].axis('off')
        
        plt.figtext(0.5, 0.01, f'IoU: {iou:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}', ha='center', fontsize=12)
        metrics = {'iou': iou, 'precision': precision, 'recall': recall}

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    plt.close(fig)
    return metrics