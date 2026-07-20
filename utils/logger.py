import logging
import os
import csv

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Sets up a logger that logs to console and optionally to a file.
    
    Args:
        name (str): Name of the logger.
        log_file (str, optional): Path to the log file.
        level (int): Logging level.
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

class TrainingLogger:
    """
    Logger for training metrics, writing to console, log file, and CSV.
    """
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, 'train.log')
        self.csv_file = os.path.join(log_dir, 'loss.csv')
        
        self.logger = setup_logger('TrainingLogger', self.log_file)
        
        # Initialize CSV
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'val_loss', 'perplexity', 'lr'])
                
    def log_metrics(self, epoch, train_loss, val_loss, perplexity, lr):
        """
        Logs training metrics for an epoch.
        """
        # Log to console and .log file
        self.logger.info(
            f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}, Perplexity = {perplexity:.4f}, LR = {lr:.6f}"
        )
        
        # Append to CSV
        with open(self.csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, perplexity, lr])
