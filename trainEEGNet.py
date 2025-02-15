import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

# Import the EEGNet model from your existing script
from EEGNet import EEGNet

# Dummy EEG Dataset (Replace with actual EEG data)
class EEGDataset(Dataset):
    def __init__(self, num_samples=1000, num_timesteps=240, num_channels=32, num_classes=4):
        self.data = torch.randn(num_samples, num_timesteps, num_channels)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label = F.one_hot(self.labels[idx], num_classes=4).float()  # Convert labels to one-hot encoding
        return self.data[idx], label

# Training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0, 0
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)  # Shape: (batch_size, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

# Evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
    return total_loss / len(dataloader), correct / len(dataloader.dataset)

if __name__ == "__main__":
    # Hyperparameters
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    num_classes = 4
    num_timesteps = 240
    num_channels = 32

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize EEGNet model
    model = EEGNet(num_temporal_filts=64, num_spatial_filts=4, num_chans=num_channels, window_length=num_timesteps, avgpool_factor=2, num_classes=num_classes)
    model.to(device)
    
    # Load data
    train_dataset = EEGDataset(num_samples=800, num_timesteps=num_timesteps, num_channels=num_channels, num_classes=num_classes)
    test_dataset = EEGDataset(num_samples=200, num_timesteps=num_timesteps, num_channels=num_channels, num_classes=num_classes)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Loss function & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
