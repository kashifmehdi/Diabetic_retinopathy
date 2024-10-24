import torch
import torch.optim as optim
import torch.nn as nn
from data_loader import get_train_loader
from model import get_model
from utils import save_checkpoint, load_checkpoint
import yaml

def train_model(config_path):
    # Load the configuration file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load the data
    train_loader = get_train_loader(config)

    # Load the model
    model = get_model(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Define optimizer, loss function, and other training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['hyperparameters']['learning_rate'])
    
    # Training loop
    num_epochs = config['hyperparameters']['epochs']
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save the model checkpoint
        save_checkpoint(model, optimizer, epoch)

if __name__ == "__main__":
    train_model('config.yaml')
