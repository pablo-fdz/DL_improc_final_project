import torch
from hw1_library.nn_model.architecture.unet import UNet  # Importing the UNet architecture

def load_model(filepath, device=None):

    """Load a model with its architecture parameters and weights. It returns the 
    loaded model in the specified or predetermined device (CPU or GPU). The model is created with the
    same parameters as the one used for training. The model weights are loaded from the
    saved state_dict.
    """ 

    # Check if a specific device is provided, otherwise use the default device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load the saved dictionary
    checkpoint = torch.load(filepath, map_location=device)
    
    # Extract the parameters and state dict
    model_params = checkpoint['model_params']
    state_dict = checkpoint['state_dict']
    
    # Create a new model with the saved parameters
    model = UNet(**model_params)
    
    # Load the weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    print(f"Loaded model from {filepath} with parameters: {model_params}")
    if 'best_val_loss' in checkpoint:
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
        
    return model