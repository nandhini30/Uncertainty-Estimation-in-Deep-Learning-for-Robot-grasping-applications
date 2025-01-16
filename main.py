import neptune

from dataloader import ImageCoordinateDataset
import albumentations as A
import albumentations.augmentations.transforms as A_transform
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from train import train_func
from model import *
import os

# Define hyperparameters
params = {
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "EPOCHS": 10,
    "LR": 0.0001,
    "batch_size": 32,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

# Define training transformations
train_transform = A.Compose([
    A_transform.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    A.Rotate(limit=45, p=0.5),
    A.VerticalFlip(p=0.2),
    A.PixelDropout(dropout_prob=0.01, p=0.3),
    A.CenterCrop(height=480, width=640, p=0.2),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy'))

# Define validation transformations
valid_transform = A.Compose([
    A_transform.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2(),
], keypoint_params=A.KeypointParams(format='xy'))

# Load image file paths (update path to your dataset directory)
image_files = [f for f in os.listdir('/path/to/your/dataset') if f.endswith('r.png')]

# Split data into training and validation sets
train_size = int(0.8 * len(image_files))
test_size = len(image_files) - train_size
tra_dataset, val_dataset = random_split(image_files, [train_size, test_size])

# Initialize datasets
train_dataset = ImageCoordinateDataset(
    '/path/to/your/dataset',  # Update to your dataset directory
    files=tra_dataset,
    train_transform=train_transform,
    valid_transform=None
)
valid_dataset = ImageCoordinateDataset(
    '/path/to/your/dataset',  # Update to your dataset directory
    files=tra_dataset,
    train_transform=None,
    valid_transform=valid_transform
)

# Initialize data loaders
train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=True)

# Initialize model and optimizer
model = GraspKeypointModel().to(params['device'])
optimizer = optim.Adam(model.parameters(), lr=params['LR'])

# Initialize Neptune logger (optional)
logger = True
if logger:
    run = neptune.init_run(
        project="your_workspace/your_project",  # Replace with your Neptune project
        api_token="YOUR_API_TOKEN_HERE",  # Replace with your API token or set as an environment variable
        tags=['scheduler', 'augmentation'],
        name='trial_with_full_transforms',
    )
    # Log hyperparameters
    run['config/hyperparameters'] = params
else:
    run = None

# Train the model
kin8nm_model, training_loss, val_loss = train_func(
    params,
    train_loader,
    valid_loader,
    valid_dataset,
    model,
    optimizer,
    scheduler=None,
    logger=run
)

# Stop Neptune run (if used)
if run:
    run.stop()
