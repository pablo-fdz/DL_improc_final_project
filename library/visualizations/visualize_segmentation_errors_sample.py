import torch
from .visualize_segmentation_errors import visualize_segmentation_errors

def visualize_segmentation_errors_sample(images, true_masks, predictions, num_classes=2, num_samples=3, threshold=0.5, rgb_bands=None):
    """
    Visualize segmentation errors for multiple random samples.
    Handles both binary and multi-class scenarios.
    
    Args:
        images (torch.Tensor): Tensor of input images [N, C, H, W].
        true_masks (torch.Tensor): Tensor of ground truth masks [N, H, W].
        predictions (torch.Tensor): Tensor of model outputs.
                                    - For binary, these are probabilities [N, H, W].
                                    - For multi-class, these are class indices [N, H, W].
        num_classes (int): Number of classes.
        num_samples (int): Number of random samples to visualize.
        threshold (float): Threshold for binary classification.
        rgb_bands (tuple): Indices for R, G, B bands for multi-band images.
    """
    if len(images) < num_samples:
        num_samples = len(images)
    indices = torch.randperm(len(images))[:num_samples].tolist()
    
    all_metrics = []
    for i, idx in enumerate(indices):
        print(f"\n--- Visualizing Sample {i+1} (Index: {idx}) ---")
        sample_metrics = visualize_segmentation_errors(
            image=images[idx], 
            true_mask=true_masks[idx], 
            pred_mask=predictions[idx], 
            num_classes=num_classes,
            threshold=threshold,
            rgb_bands=rgb_bands
        )
        all_metrics.append(sample_metrics)
    
    if all_metrics:
        avg_metrics = {key: sum(m[key] for m in all_metrics) / len(all_metrics) for key in all_metrics[0]}
        print("\n--- Average Metrics Across Visualized Samples ---")
        for key, value in avg_metrics.items():
            print(f"Average {key.capitalize()}: {value:.4f}")