import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_loss_curves(csv_path='logs/loss.csv', save_path=None):
    """
    Plots training and validation loss curves from a CSV file.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    if 'val_loss' in df.columns and not df['val_loss'].isna().all():
        plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
        
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_lr_schedule(lrs, save_path=None):
    """
    Plots the learning rate schedule over time.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(lrs)), lrs, label='Learning Rate')
    plt.xlabel('Step/Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_attention_weights(attention_weights, save_path=None):
    """
    Plots a heatmap of attention weights.
    
    Args:
        attention_weights (np.ndarray or torch.Tensor): Attention weights of shape (seq_len, seq_len).
    """
    if hasattr(attention_weights, 'detach'):
        attention_weights = attention_weights.detach().cpu().numpy()
        
    plt.figure(figsize=(8, 8))
    plt.imshow(attention_weights, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.xlabel('Key Positions')
    plt.ylabel('Query Positions')
    plt.title('Attention Weights Heatmap')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
