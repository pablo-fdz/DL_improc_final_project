import torch
from tqdm import tqdm

def predict(model, dataloader, device, return_probs=True, threshold=0.5):
    
    """
    Predict the output for a given dataloader using the provided model.
    
    Args:
        model: The trained model
        dataloader: DataLoader for the dataset to predict
        device: Device to perform the computation on (CPU or GPU)
        return_probs: Whether to return probabilities or binary predictions
        threshold: Threshold for binary predictions
        
    Returns:
        predictions: List of predictions (probabilities or binary masks)
        true_labels: List of true labels (if available in dataloader)
    """
    
    model.eval()  # Set model to evaluation mode (using it for inference)
    predictions = []
    true_labels = []
    
    with torch.no_grad():  # Disable gradient calculation (saves memory and computation)
        for inputs, labels in tqdm(dataloader, desc="Generating predictions"):
            inputs = inputs.to(device)  # Move data to device
            
            # Forward pass
            logits = model(inputs)

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits.squeeze(1))
            
            if return_probs:
                # Return probabilities
                preds = probs
            else:
                # Return binary predictions
                preds = (probs > threshold).float()
            
            # Store predictions and true labels
            predictions.append(preds) 
            true_labels.append(labels.squeeze(1))
    
    # Concatenate batch predictions into a single tensor
    predictions = torch.cat(predictions, dim=0)
    true_labels = torch.cat(true_labels, dim=0)
    
    return predictions, true_labels