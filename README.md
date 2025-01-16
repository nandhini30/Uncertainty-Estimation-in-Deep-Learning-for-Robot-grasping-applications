# Uncertainty Estimation in Deep Learning for Robot Grasping Applications

Understanding uncertainty in robotic grasping is akin to giving robots a sixth sense. It enables them to adapt to unpredictable situations, make informed decisions, and ensure robust performance in real-world applications. This project demonstrates the significance of uncertainty estimation in enhancing the reliability of robotic systems, with experiments conducted on UCI regression 2D datasets and the Cornell Grasping Dataset in a 3D domain.

## Abstract

Our research focuses on comparing Uncertainty Estimation methods in robotic grasping tasks. Starting with UCI regression 2D datasets, we transitioned to the Cornell Grasping Dataset and further enriched it with a **manually collected dataset containing annotated keypoints**. We evaluated the following five loss functions for uncertainty estimation:

1. **Gaussian Negative Log-Likelihood**
2. **Laplace Negative Log-Likelihood**
3. **Cauchy Negative Log-Likelihood**
4. **Evidential Loss**
5. **Generalized Gaussian Negative Log-Likelihood** (Newly Introduced loss function in this work)

The neural networks were modeled to predict values along with their uncertainty. Finally, the results were implemented on a Kinova robot arm, demonstrating that the **Generalized Gaussian Negative Log-Likelihood** excels in predicting accurate grasp keypoints with superior uncertainty estimates.

---

## Dataset Information

### 1. Cornell Grasping Dataset
The Cornell Grasping Dataset is a standard benchmark for robotic grasping tasks. It includes 3D images of objects with labeled grasp rectangles.

- **Dataset Link**: [Cornell Grasping Dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp)

### 2. Manually Collected Dataset
To enhance the dataset, we collected additional images of objects of varying shapes and sizes and annotated them with **eight keypoints** representing optimal grasp positions.

- **Annotations**: Eight keypoints labeled per object for grasping.
- **Integration**: This dataset is merged with the Cornell Grasping Dataset for training and validation.

---

## Contributions

1. **Uncertainty Estimation in Robotics**:
   - Systematic comparison of Uncertainty Estimation methods using 2D and 3D datasets.
2. **Enhanced Dataset**:
   - Integration of manually collected data with the Cornell Grasping Dataset.
3. **Neural Network Design**:
   - Modeling networks to output predictions and uncertainties.
4. **Loss Function Evaluation**:
   - In-depth analysis of five loss functions for uncertainty estimation.
5. **Practical Implementation**:
   - Real-world validation using a Kinova robot arm.

---

## Installation
