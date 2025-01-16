import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ImageCoordinateDataset(Dataset):
    """
    A custom Dataset class to handle images and their corresponding coordinate data.
    
    Attributes:
        root_dir (str): Root directory containing image and coordinate files.
        image_files (list): List of image file names.
        scaler (StandardScaler): Scaler for normalizing coordinate data.
        train_transform (callable, optional): Transformation pipeline for training data.
        valid_transform (callable, optional): Transformation pipeline for validation data.
    """

    def __init__(self, root_dir, files, train_transform=None, valid_transform=None):
        """
        Initializes the dataset with image file paths, transformations, and a scaler.

        Args:
            root_dir (str): Path to the root directory containing images and coordinates.
            files (list): List of file names for images.
            train_transform (callable, optional): Transformations for training.
            valid_transform (callable, optional): Transformations for validation.
        """
        self.root_dir = root_dir
        self.image_files = files

        # Initialize scaler with a predefined range.
        self.scaler = StandardScaler()
        self.scaler.fit([[640, 480], [1, 1]])

        self.train_transform = train_transform
        self.valid_transform = valid_transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Retrieves a sample (image and coordinates) from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: Transformed image tensor and normalized coordinates tensor.
        """
        # Construct file paths for the image and its corresponding coordinate file.
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_file)
        coordinate_file = image_file.replace('r.png', 'cpos.txt')
        coordinate_path = os.path.join(self.root_dir, coordinate_file)

        # Load the image and convert it to RGB format.
        image = Image.open(image_path).convert('RGB')

        # Load and preprocess coordinates.
        coordinate = self._load_coordinate(coordinate_path)

        if self.train_transform is not None:
            # Apply transformations for training data.
            transformed = self.train_transform(image=np.array(image), keypoints=coordinate)
            transformed_image = transformed['image']
            transformed_keypoints = transformed['keypoints']

            # Normalize coordinates using the scaler.
            transformed_keypoints = torch.tensor(transformed_keypoints, dtype=torch.float32)
            coordinates = self.scaler.transform(transformed_keypoints)
            image_tensor = np.transpose(transformed_image, (2, 0, 1))
        elif self.valid_transform is not None:
            # Apply transformations for validation data.
            transformed = self.valid_transform(image=np.array(image))
            transformed_image = transformed['image']
            coordinates = self.scaler.transform(coordinate)
            image_tensor = np.transpose(transformed_image, (2, 0, 1))
        else:
            # No transformations applied.
            image_tensor = np.array(image).transpose((2, 0, 1))
            coordinates = self.scaler.transform(coordinate)

        return image_tensor, coordinates

    def _load_coordinate(self, coordinate_path):
        """
        Loads coordinate data from a file.

        Args:
            coordinate_path (str): Path to the coordinate file.

        Returns:
            torch.Tensor: Tensor containing coordinate points.
        """
        with open(coordinate_path, 'r') as f:
            lines = f.readlines()

        coordinates = []
        for line in lines[:8]:  # Limit to the first 8 lines.
            x, y = map(float, line.strip().split())
            coordinates.append([x, y])

        return torch.tensor(coordinates)
