# Diabetic Retinopathy Detection 👁️

An automated deep learning system for detecting diabetic retinopathy using PyTorch and EfficientNetB3 architecture. This project achieves 97.98% accuracy in detecting retinopathy from retinal images, making it a reliable tool for medical professionals in early diagnosis and treatment planning.

## 🎯 Project Overview

This project implements a deep learning model to detect diabetic retinopathy from retinal images. Diabetic retinopathy is a diabetes complication that affects eyes and is a leading cause of blindness. Early detection is crucial for effective treatment.

### What is Diabetic Retinopathy? 🔬

Diabetic retinopathy is a diabetes-related eye condition that occurs when high blood sugar levels damage the blood vessels in the retina. The condition progresses through several stages:

0. **No DR**: Healthy retina with no apparent changes
1. **Mild NPDR**: Small areas of balloon-like swelling in the retina's blood vessels
2. **Moderate NPDR**: Some blood vessels that nourish the retina become blocked
3. **Severe NPDR**: Many blood vessels are blocked, depriving blood supply to the retina
4. **PDR**: The most advanced stage where new, abnormal blood vessels grow

### Why Deep Learning for Detection? 🤖

Traditional detection methods rely heavily on experienced ophthalmologists manually reviewing retinal images, which is:
- Time-consuming ⏰
- Expensive 💰
- Subject to human error and fatigue 😓
- Not readily available in many parts of the world 🌍
  
## Model Architecture 🏗️

I have utilize EfficientNetB3, a state-of-the-art convolutional neural network known for its:
- Excellent balance of accuracy and computational efficiency
- Strong feature extraction capabilities
- Proven track record in medical image analysis

This project represents a significant step forward in the application of artificial intelligence to medical diagnostics, combining cutting-edge deep learning techniques with practical clinical requirements to create a tool that can make a real difference in preventing vision loss due to diabetic retinopathy.

## Features ✨ 

- High accuracy detection (97.98% on test set)
- Built with PyTorch and EfficientNetB3
- Automated training pipeline
- Comprehensive data preprocessing
- Model checkpointing and evaluation
- Training progress monitoring

## Project Goals 🎯 

1. **Early Detection** 🔍
   - Develop an automated system for early detection of diabetic retinopathy
   - Reduce the time and cost associated with manual screening
   - Enable rapid diagnosis in resource-limited settings

2. **Accuracy & Reliability** ✅
   - Achieve high accuracy in classification (currently at 97.98%)
   - Minimize false negatives to ensure patient safety
   - Provide consistent and reliable results

3. **Clinical Support** 🏥
   - Assist healthcare professionals in diagnosis
   - Provide a tool for mass screening programs
   - Support early intervention decisions

4. **Accessibility** 🌍
   - Create a solution that can be deployed in various healthcare settings
   - Minimize hardware requirements while maintaining performance
   - Enable easy integration with existing healthcare systems

## Configuration 🔧
The project uses a YAML configuration file (`config.yaml`) to manage all parameters:

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

## Project Structure 📁

```
Diabetic Retinopathy/
│
├── data/
│   ├── test_images/                # Directory containing test images
│   ├── train_images/               # Directory containing training images
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
└── requirements.txt                # Python dependencies for the project
```

## Getting Started 🚀

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch 1.8+

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diabetic-retinopathy.git
cd diabetic-retinopathy
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage 💻

1. Prepare your data:
   - Place training images in `data/train_images/`
   - Place test images in `data/test_images/`
   - Ensure CSV files are properly formatted

2. Configure the model:
   - Adjust parameters in `config.yaml`

3. Train the model:
```bash
python src/train.py
```

4. Evaluate the model:
```bash
python src/evaluate.py
```

## Model Performance 📈

- Training Progress:
  - Starting Accuracy: 76.16%
  - Final Accuracy: 97.05%
  - Loss Reduction: 0.6411 → 0.0896

- Test Accuracy: 97.98%

## Project Components 🏗️

- `data_loader.py`: Handles data preprocessing and loading
- `model.py`: Contains EfficientNetB3 model architecture
- `train.py`: Training pipeline implementation
- `evaluate.py`: Model evaluation scripts
- `utils.py`: Utility functions
- `EDA.ipynb`: Exploratory Data Analysis notebook

## Dataset 📊

The dataset consists of retinal images categorized for diabetic retinopathy detection. Images are organized in the following structure:
- Training images: Located in `data/train_images/`
- Test images: Located in `data/test_images/`
- Corresponding CSV files contain labels and metadata

## Technical Details 🛠️

- **Framework**: PyTorch
- **Model Architecture**: EfficientNetB3
- **Training Parameters**:
  - Epochs: 10
  - Batch Size: 32
  - Learning Rate: 0.001
  - Input Image Size: 224x224
  - Number of Classes: 5
  - Pre-trained Weights: ImageNet

### Future Developments 🚀

Planned enhancements include:
- Multi-disease detection capabilities
- Real-time processing features
- Mobile deployment options
- Integration with electronic health records
- Enhanced visualization tools for medical professionals

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

[Your contact information]
