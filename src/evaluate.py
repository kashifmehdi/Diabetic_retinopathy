import torch
from data_loader import get_train_loader
from model import get_model
from utils import load_checkpoint
import yaml

def evaluate_model(config_path):
    # Load the configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load the data
    test_loader = get_train_loader(config)

    # Load the model
    model = get_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Load the best model checkpoint
    load_checkpoint(model, 'best_model.pth')

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    evaluate_model('config.yaml')
