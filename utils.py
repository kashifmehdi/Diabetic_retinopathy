import torch

def save_checkpoint(model, optimizer, epoch, filename="best_model.pth"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, filename)
    print(f'Checkpoint saved at epoch {epoch}')

def load_checkpoint(model, filename="best_model.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Checkpoint loaded from {filename}')
