from dataloader import *
from model import *
from loss import *
from metrics import *
from config import *
import torch
import pandas as pd
import scipy.stats as stats
from scipy.special import gamma
import numpy as np
import torch.optim as optim

def test_validation(model, valid_loader):
    """
    Validate the model on the validation dataset and compute predictions, variance, and errors.

    Args:
        model (torch.nn.Module): Trained model to evaluate.
        valid_loader (DataLoader): DataLoader for validation data.

    Returns:
        tuple: Outputs, variance, validation differences, and true values.
    """
    model.eval()

    val_difference = []
    out = []
    variance = []
    inputs = []

    with torch.no_grad():
        for i, (X_val_batch, y_val_batch) in enumerate(valid_loader):
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
            y_val_batch = y_val_batch[:, :, 0]

            y_val_pred, var = model(X_val_batch)

            std = np.sqrt(var.cpu().numpy())
            variance.extend(var.cpu().numpy())

            val_loss = criterion(y_val_pred, y_val_batch, var)
            val_difference.extend((y_val_batch - y_val_pred).tolist())

            norm_dist = stats.norm(loc=y_val_batch.cpu().numpy(), scale=std)
            l, u = norm_dist.interval(alpha=0.95)
            Int_score = np.average(interval_score(y_val_batch.cpu().numpy(), l, u))

            inputs.extend(y_val_batch.cpu().numpy())
            out.extend(y_val_pred.cpu().numpy().tolist())

    return out, variance, val_difference, inputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraspKeypointModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

# Replace with appropriate path to the saved model
val_model = torch.load("path/to/your/model.pth")  # Update the path

output, var, validation_difference, true_value = test_validation(val_model, valid_loader)
