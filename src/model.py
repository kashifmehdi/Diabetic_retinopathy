import torch
import torch.nn as nn
from torchvision import models

def get_model(config):
    model = models.efficientnet_b3(pretrained=config['model']['weights'] == 'imagenet')
    
    # Modify the classifier to match the number of classes
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(num_features, config['hyperparameters']['num_classes'])
    )
    
    return model
