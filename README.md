# Diabetic Retinopathy Detection
Deep learning model for detecting Diabetic retinopathy using EfficientNet and Pytorch

## Setup
1. Create Virtual Environment
2. Install requirements: `pip install -r requirements.txt`
3. Place raw images in `data/`
4. Run preprocessing scripts
5. Train model : `python train.py`
6. Evaluate Model : `python evaluate.py`

## Project Structure
# Diabetic Retinopathy Detection Project

This repository contains code for a Diabetic Retinopathy detection project using the EfficientNetB3 model in PyTorch. The project is structured to facilitate easy dataset management, training, and evaluation.

## File Structure

```bash
Diabetic Retinopathy/
│
├── data/
│   ├── test_images/                # Directory containing test images
│   ├── train_images/               # Directory containing training images
│   ├── sample_submission.csv       # Sample submission CSV (for Kaggle-style competitions)
│   ├── test.csv                    # CSV file containing test image names and metadata
│   ├── train.csv                   # CSV file containing training image names and labels
│
├── logs/                           # Directory for storing training logs (created during training)
│
├── notebook/
│   └── EDA.ipynb                   # Jupyter notebook for exploratory data analysis (EDA)
│
├── src/                            # Source code for training, evaluating, and data loading
│   ├── __pycache__/                # Cache directory for compiled Python files (auto-generated)
│   ├── __init__.py                 # Init file for the source code package
│   ├── data_loader.py              # Data loader for loading and transforming images
│   ├── evaluate.py                 # Script to evaluate the model on test data
│   ├── model.py                    # Model architecture using EfficientNetB3
│   ├── utils.py                    # Utility functions like checkpoint saving/loading
│   └── train.py                    # Script to train the model
│
├── venv/                           # Python virtual environment (contains all installed dependencies)
│
├── config.yaml                     # YAML configuration file for the project
├── README.md                       # Project documentation (this file)
├── requirements.txt                # Python dependencies for the project
└── train.py                        # Main script to start model training



## Configuration 
Model and training parameters can be modified in `config.yaml`
```

Setup Instructions
1. Clone the repository
```bash
git clone https://github.com/yourusername/diabetic-retinopathy.git
cd diabetic-retinopathy
```

2. Create a Virtual Environment
You can create a virtual environment using Python's venv:
```bash
python -m venv venv
```

Activate the virtual environment:

 - On Windows:
```bash
venv\Scripts\activate
```

 - On macOS/Linux:
```bash
source venv/bin/activate
```

3. Install Dependencies
Install the required Python packages using pip:
```bash
pip install -r requirements.txt
```

4. Configuration
The project configuration is controlled by the config.yaml file. It contains paths to the dataset, hyperparameters, and model architecture. Adjust the file as needed:

```yaml
# Example configuration

data:
  train_csv: 'data/train.csv'           # Path to the training CSV file
  train_images_dir: 'data/train_images' # Directory containing training images
  test_images_dir: 'data/test_images'   # Directory containing test images

hyperparameters:
  batch_size: 32                        # Batch size for training
  epochs: 10                            # Number of training epochs
  learning_rate: 0.001                  # Learning rate for the optimizer
  image_size:
    width: 224                          # Width of input images
    height: 224                         # Height of input images
  num_classes: 5                        # Number of classification labels

model:
  architecture: 'EfficientNetB3'        # Model architecture to use
  weights: 'imagenet'                   # Pre-trained weights to use
```

5. Train the Model
Run the training script:

```bash
python src/train.py
```

6. Evaluate the Model
Run the evaluation script to test the model on your dataset:
```bash
python src/evaluate.py
```

7. Jupyter Notebook for EDA
The EDA.ipynb notebook in the notebook directory contains exploratory data analysis (EDA) code. You can use it to better understand the dataset before training the model.

8. Logs
Training logs will be saved in the logs/ directory.



