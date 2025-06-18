import torch
from .visualize_segmentation_errors import visualize_segmentation_errors

def visualize_segmentation_errors_sample(images, true_masks, pred_probs, num_samples=3, threshold=0.5):
    """
    Visualize segmentation errors for multiple random samples.
    
    Args:
        images: Tensor of input images [N, C, H, W]
        true_masks: Tensor of ground truth masks [N, H, W]
        pred_probs: Tensor of predicted probabilities [N, H, W]
        num_samples: Number of random samples to visualize
        threshold: Threshold for converting probabilities to binary masks
    """
    # Get random indices
    indices = torch.randperm(len(images))[:num_samples].tolist()
    
    # Visualize each sample
    metrics = []
    for idx in indices:
        img = images[idx]
        true_mask = true_masks[idx]
        pred_prob = pred_probs[idx]
        
        print(f"\nSample {idx}:")
        sample_metrics = visualize_segmentation_errors(img, true_mask, pred_prob, threshold)
        metrics.append(sample_metrics)
    
    # Print average metrics
    avg_metrics = {key: sum(m[key] for m in metrics)/len(metrics) for key in metrics[0] 
                  if key in ['precision', 'recall', 'f1', 'iou']}
    print("\nAverage metrics across samples:")
    for key, value in avg_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")