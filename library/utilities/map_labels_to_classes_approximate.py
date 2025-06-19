import torch

# Alternative approach if you want to handle approximate values (rounded to 2 decimals)
def map_labels_to_classes_approximate(labels_tensor):
    """
    Map normalized label values (0-1) to class indices (0-4).
    Handles approximate values by finding the closest class
    """
    # Convert to float for calculations
    labels_float = labels_tensor.float()
    
    # Define the normalized class centers
    class_centers = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])  # Corresponding to 0, 64, 128, 191, 255
    
    # Find the closest class for each pixel
    labels_mapped = torch.zeros_like(labels_tensor, dtype=torch.long)
    
    for i, center in enumerate(class_centers):
        if i == 0:
            mask = labels_float <= (class_centers[0] + class_centers[1]) / 2
        elif i == len(class_centers) - 1:
            mask = labels_float > (class_centers[i-1] + class_centers[i]) / 2
        else:
            mask = (labels_float > (class_centers[i-1] + class_centers[i]) / 2) & \
                   (labels_float <= (class_centers[i] + class_centers[i+1]) / 2)
        
        labels_mapped[mask] = i
    
    return labels_mapped