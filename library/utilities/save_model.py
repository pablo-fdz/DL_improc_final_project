import torch
import os

# Save both model weights and architecture parameters
def save_model(model, filename, best_val_loss, **model_params):
    
    """Save model weights and architecture parameters"""
    
    try:

        save_dict = {
            'state_dict': model.state_dict(),  # model weights
            'model_params': model_params,  # model architecture parameters
            'best_val_loss': best_val_loss,  # best validation loss
        }
        model_path = os.path.join('models/', f'{filename}_val_loss_{best_val_loss:.4f}.pth')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Create directory if it doesn't exist
        torch.save(save_dict, model_path)
        print(f"Model saved to {model_path}")
    
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return None