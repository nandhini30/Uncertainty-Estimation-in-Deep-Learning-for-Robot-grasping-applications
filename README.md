# Uncertainty-Estimation-in-Deep-Learning-for-Robot-grasping-applications


```markdown
# Robot Grasping with Uncertainty Estimation

In the real-world robot grasping applications, understanding uncertainty is akin to giving robots a sixth sense, helping them adapt to unpredictable situations and make better choices. This research demonstrates the significance of Uncertainty Estimation in enhancing the reliability of robotic systems, starting with a comparison of methods on UCI regression 2D datasets and progressing to the Cornell Grasping Dataset for complex 3D domains.

## Abstract

Understanding uncertainty helps robots make informed decisions, ensuring robust performance even when faced with unpredictable situations, ultimately enhancing the reliability of tasks. Our research compares Uncertainty Estimation methods using UCI regression 2D datasets before transitioning to the Cornell Grasping Dataset in a 3D domain. We also incorporated a **manually collected dataset with annotated keypoints**, adding these to the existing dataset to enrich the diversity of training data. We employ five distinct loss functions:

1. **Gaussian Negative Log-Likelihood**
2. **Laplace Negative Log-Likelihood**
3. **Cauchy Negative Log-Likelihood**
4. **Evidential Loss**
5. **Generalized Gaussian Negative Log-Likelihood**

Our approach involves neural networks that output predictions along with uncertainty estimates. The findings were implemented on a real-time Kinova robot arm, demonstrating that the **Generalized Gaussian Negative Log-Likelihood** excels in predicting accurate keypoints with better uncertainty estimates, making it highly effective for real-world robotic grasping tasks.

---

## Dataset Information

### 1. Cornell Grasping Dataset
The Cornell Grasping Dataset is a standard benchmark dataset for robotic grasping tasks, containing 3D images of objects along with labeled grasp rectangles.

- **Link**: [Cornell Grasping Dataset](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp)

### 2. Manually Collected Dataset
We collected an additional dataset with objects of various shapes and sizes, manually annotating the grasp keypoints. This dataset was merged with the Cornell Grasping Dataset to enhance the diversity and robustness of the training data.

- **Annotations**: Eight Keypoints were labeled for each object to represent optimal grasp positions.
- **Integration**: This dataset is used alongside the Cornell dataset in the training pipeline.

---

## Contributions

1. **Uncertainty Estimation in Robotic Grasping**:
   - A systematic comparison of Uncertainty Estimation methods using both 2D and 3D datasets.
2. **Enhanced Dataset**:
   - Integration of a manually collected dataset with annotated keypoints into the Cornell Grasping Dataset.
3. **Neural Network Design**:
   - Modeling networks to output predictions with uncertainty.
4. **Loss Function Evaluation**:
   - Detailed analysis of five distinct loss functions for Uncertainty Estimation.
5. **Real-World Implementation**:
   - Translating findings to practical use on a Kinova robot arm.

---

## Installation

### Prerequisites
1. Python 3.8 or higher
2. CUDA-enabled GPU (optional but recommended)

### Install Dependencies
Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## Running the Code

### Dataset Preparation
1. Download the **Cornell Grasping Dataset** from [here](https://www.kaggle.com/datasets/oneoneliu/cornell-grasp).
2. Place your manually collected dataset in the same directory as the Cornell dataset and ensure the keypoints are properly labeled.
3. Update dataset paths in `config.py` to match your local directory structure.

### Running the Script
The primary script for running the experiments is `main.py`. Modify the script to choose the desired loss function and configuration.

```bash
python main.py
```

- **Configuration**: Update the `config.py` file to set the following:
  - Learning rate
  - Batch size
  - Number of epochs
  - Loss function
  - Device (CPU or GPU)

### Testing
To test a trained model:

```bash
python test.py
```

This will output predictions, variance estimates, and interval scores for evaluation.

---

## Results

Our experiments demonstrate that **Generalized Gaussian Negative Log-Likelihood** provides superior performance in predicting accurate keypoints with better uncertainty estimates, proving its efficacy for real-world robotic grasping tasks.

---

## Conclusion

In robot grasping applications, uncertainty estimation improves adaptability and decision-making in uncertain environments. By incorporating manually collected data with annotated keypoints into the Cornell Grasping Dataset and leveraging advanced loss functions, this research demonstrates that **Generalized Gaussian Negative Log-Likelihood** is the most effective method for grasp keypoint prediction with uncertainty estimation.

---


## Acknowledgments

We thank the creators of the Cornell Grasping Dataset and acknowledge the efforts in manually collecting and annotating additional data.
```

### Key Updates:
1. **Dataset Information**:
   - Included details about the Cornell Grasping Dataset and your manually collected dataset with annotations.
   - Explained how the datasets were integrated.

2. **Instructions**:
   - Added clear steps for preparing the dataset and updating paths in the code.

3. **Focus on `main.py`**:
   - Reinforced that `main.py` is the entry point for training experiments.

Let me know if additional sections or modifications are needed!
