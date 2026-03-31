import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import ResNet50_Weights

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
data_dir = './archive/Testing'  # Path to your test data
model_path = 'brain_tumor_resnet50_final.pth'  # Path to the saved model

# Load default weights and preprocessing for ResNet50
weights = ResNet50_Weights.DEFAULT

# Data transforms (must be the same as during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    weights.transforms()  # Preprocessing from ResNet50 weights
])

# Load dataset
test_dataset = datasets.ImageFolder(data_dir, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Print dataset info
print(f"Number of test images: {len(test_dataset)}")
print(f"Class names: {test_dataset.classes}")

# Define the model with the same architecture as in the training script
model = models.resnet50(weights=weights)
num_classes = len(test_dataset.classes)

# Matching the model architecture used in training (using Dropout in the final layer)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout for regularization
    nn.Linear(model.fc.in_features, num_classes)
)

# Load the trained model
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)

# Evaluate the model
model.eval()  # Set the model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No need to compute gradients for testing
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Accuracy
accuracy = 100 * correct / total
print(f"Accuracy on the test dataset: {accuracy:.2f}%")
