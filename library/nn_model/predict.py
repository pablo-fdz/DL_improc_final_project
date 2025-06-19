import torch
from tqdm import tqdm

def predict(model, dataloader, device, num_classes, return_probs=True, threshold=0.5):
    
    """
    Predict the output for a given dataloader using the provided model.
    Handles both binary and multi-class segmentation.
    
    Args:
        model: The trained model.
        dataloader: DataLoader for the dataset to predict.
        device: Device to perform the computation on (e.g., 'cuda' or 'cpu').
        num_classes: The number of classes for segmentation.
        return_probs: If True, returns probabilities.
                      - For binary (num_classes=2), returns sigmoid probabilities [B, H, W].
                      - For multi-class, returns softmax probabilities [B, C, H, W].
                      If False, returns predicted class indices.
                      - For binary, returns a binary mask (0 or 1) [B, H, W].
                      - For multi-class, returns class indices (0 to C-1) [B, H, W].
        threshold: Threshold for converting probabilities to binary masks in the binary case.
                   Ignored for multi-class classification.
        
    Returns:
        predictions: A tensor of predictions (probabilities or class masks).
        true_labels: A tensor of the corresponding true labels.
    """
    
    model.eval()  # Set model to evaluation mode (using it for inference)
    predictions = []
    true_labels = []
    
    with torch.no_grad():  # Disable gradient calculation (saves memory and computation)
        for inputs, labels in tqdm(dataloader, desc="Generating predictions"):
            inputs = inputs.to(device)  # Move data to device
            
            # Forward pass
            logits = model(inputs)

            if num_classes > 2:
                # --- Multi-class classification ---
                if return_probs:
                    # Apply softmax to get probabilities across classes
                    preds = torch.softmax(logits, dim=1)
                else:
                    # Get the predicted class index by finding the max logit
                    preds = torch.argmax(logits, dim=1)
            elif num_classes == 2:
                # --- Binary classification ---
                # Squeeze channel dimension if it exists (from [B, 1, H, W] to [B, H, W])
                if logits.dim() == 4 and logits.shape[1] == 1:
                    logits = logits.squeeze(1)
                
                # Apply sigmoid to convert logits to probabilities
                probs = torch.sigmoid(logits)
                
                if return_probs:
                    preds = probs
                else:
                    # Return binary predictions based on threshold
                    preds = (probs > threshold).float()
            else:
                raise ValueError("num_classes must be 2 (binary) or greater than 2 (multi-class).")
            
            # Store predictions and true labels on CPU to free up GPU memory
            predictions.append(preds.cpu()) 
            
            # Squeeze labels if they have a channel dim of 1
            if labels.dim() == 4 and labels.shape[1] == 1:
                labels = labels.squeeze(1)
            true_labels.append(labels.cpu())
    
    # Concatenate all batch predictions into a single tensor
    predictions = torch.cat(predictions, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    
    return predictions, true_labels