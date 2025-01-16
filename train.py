from model import *
from loss import *
import torch.optim as optim
import matplotlib.pyplot as plt
from config import *
import numpy as np
import albumentations as A

def train_func(params, train_loader, valid_loader, valid_dataset, model, optimizer, logger=None):
    """
    Train the model using the specified parameters, data loaders, and optimizer.

    Args:
        params (dict): Training configuration parameters (e.g., epochs, device).
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        valid_dataset (Dataset): Validation dataset.
        model (torch.nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        logger (neptune.Logger, optional): Logger for tracking training progress.

    Returns:
        tuple: Trained model, list of training losses, list of validation losses.
    """
    last_loss = float('inf')
    patience = 5
    trigger_times = 0
    model_save_loc = "path/to/your/model_save_location.pth"  # Replace with actual path

    print('Training started...')
    Train_epoch_loss = []
    Val_epoch_loss = []

    for e in range(params['EPOCHS']):
        model.train()
        train_running_loss = []

        for i, (X_train_batch, y_train_batch) in enumerate(train_loader):
            X_train_batch, y_train_batch = X_train_batch.to(params['device']), y_train_batch.to(params['device'])

            optimizer.zero_grad()
            outputs, var = model(X_train_batch)

            train_loss = criterion(y_train_batch, outputs, var)
            train_running_loss.extend(train_loss.tolist())

            train_loss.mean().backward()
            optimizer.step()

        train_loss = np.average(train_running_loss)
        Train_epoch_loss.append(train_loss)

        model.eval()
        with torch.no_grad():
            valid_running_loss = []
            val_difference = []

            for i, (X_val_batch, y_val_batch) in enumerate(valid_loader):
                X_val_batch, y_val_batch = X_val_batch.to(params['device']), y_val_batch.to(params['device'])
                y_val_batch = y_val_batch[:, :, 0]

                y_val_pred, var = model(X_val_batch)

                val_loss = criterion(y_val_pred, y_val_batch, var)
                valid_running_loss.extend(val_loss.tolist())
                val_difference.extend((y_val_batch - y_val_pred).tolist())

            val_loss = np.average(valid_running_loss)

        print(f"Epoch: {e+1}/{params['EPOCHS']}.. Training Loss: {train_loss:.3f}.. Validation Loss: {val_loss:.3f}")

        if logger is not None:
            logger['plots/training/train_loss'].log(train_loss)
            logger['plots/training/validation_loss'].log(val_loss)

        if val_loss > last_loss:
            trigger_times += 1
            print(f'Trigger Times: {trigger_times}\n')

            if trigger_times >= patience:
                print('Early stopping! Stopping training.')
                break
        else:
            print('Trigger times: 0\n')
            trigger_times = 0
            torch.save(model, model_save_loc)

        last_loss = val_loss

    plt.figure(figsize=(10, 7))
    plt.plot(Train_epoch_loss, color='orange', label='Training Loss')
    plt.plot(Val_epoch_loss, color='red', label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('path/to/your/loss_plot.png')  # Replace with actual path
    plt.show()

    print('Training complete.')
    return model, Train_epoch_loss, Val_epoch_loss
