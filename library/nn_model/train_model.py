from torchmetrics import classification
from tqdm import tqdm
import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, patience=10, delta=0.0001):
    """
    Train the model with early stopping based on validation loss.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Maximum number of training epochs
        patience: Number of epochs to wait for improvement before stopping
        delta: Minimum change in validation loss to qualify as improvement
        
    Returns:
        model: Trained model
        history: Dictionary containing training and validation metrics
    """
    
    ########################################
    # 1. Function setup and initialization
    ########################################

    # Initialize variables for early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_weights = None
    
    # For tracking metrics
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    # Track total batches for progress display
    total_train_batches = len(train_loader)
    total_val_batches = len(val_loader)
    
    # Initialize metrics for IoU (Intersection over Union) calculation
    # - it calculates the overlap between the predicted and ground truth masks
    device = next(model.parameters()).device  # Get model's device
    iou_metric = classification.BinaryJaccardIndex().to(device)
    
    ########################################
    # 2. Training loop
    ########################################

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print('-' * 50)
        
        ####################
        # 2.1. Training phase
        ####################

        model.train()  # Set model to training mode
        running_train_loss = 0.0
        running_train_iou = 0.0
        
        # Use tqdm for progress bar during training
        progress_bar = tqdm(enumerate(train_loader), total=total_train_batches)
        for batch_idx, (inputs, labels) in progress_bar:
            # Move data to device
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Squeeze to match dimensions for loss calculation
            outputs = outputs.squeeze(1)  # From [B, 1, H, W] to [B, H, W]
            labels = labels.squeeze(1)    # From [B, 1, H, W] to [B, H, W]
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_train_loss += loss.item() * inputs.size(0)
            
            # Calculate IoU (convert outputs to binary predictions using 0.5 threshold)
            
            # Calculation of the IoU metric if the model outputs logits
            probs = torch.sigmoid(outputs)  # Conversion to probabilities - needed if the model outputs logits (comment otherwise)
            preds = (probs > 0.5).float()  # Now threshold the probabilities
            running_train_iou += iou_metric(preds, labels) * inputs.size(0)
            
            # Update progress bar
            progress_bar.set_description(f"Train Loss: {loss.item():.4f}")
        
        # Calculate epoch statistics
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        epoch_train_iou = running_train_iou / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        history['train_metrics'].append(epoch_train_iou)
        
        ####################
        # 2.2. Validation phase (no gradient computation, backpropagation or weight updates)
        ####################

        model.eval()  # Set model to evaluation mode
        running_val_loss = 0.0
        running_val_iou = 0.0
        
        # No gradient computation during validation
        with torch.no_grad():  # Disable gradient calculation
            progress_bar = tqdm(enumerate(val_loader), total=total_val_batches)
            for batch_idx, (inputs, labels) in progress_bar:
                # Move data to device
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Squeeze to match dimensions
                outputs = outputs.squeeze(1)
                labels = labels.squeeze(1)
                
                # Compute loss
                loss = criterion(outputs, labels)
                
                # Update statistics
                running_val_loss += loss.item() * inputs.size(0)
                
                # Calculate IoU
                probs = torch.sigmoid(outputs)  # Conversion to probabilities - needed if the model outputs logits (comment otherwise)
                preds = (outputs > 0.5).float()
                running_val_iou += iou_metric(preds, labels) * inputs.size(0)
                
                # Update progress bar
                progress_bar.set_description(f"Val Loss: {loss.item():.4f}")
        
        # Calculate epoch statistics
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_iou = running_val_iou / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        history['val_metrics'].append(epoch_val_iou)
        
        # Print epoch statistics
        print(f"Training Loss: {epoch_train_loss:.4f}, IoU: {epoch_train_iou:.4f}")
        print(f"Validation Loss: {epoch_val_loss:.4f}, IoU: {epoch_val_iou:.4f}")
        
        ####################
        # 2.3. Early stopping check
        ####################

        # Check for improvement in validation loss
        if epoch_val_loss < best_val_loss - delta:
            print(f"Validation loss improved from {best_val_loss:.4f} to {epoch_val_loss:.4f}\n")
            best_val_loss = epoch_val_loss
            best_model_weights = model.state_dict().copy()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            print(f"No improvement in validation loss for {early_stop_counter} epochs\n")
            
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs\n")
                break
    
    ########################################
    # 3. Final model loading: saving weights
    ########################################

    # Load best model weights
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
        print(f"Loaded best model weights with validation loss: {best_val_loss:.4f}")

    return model, history, best_val_loss