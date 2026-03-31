import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torchvision.models import ResNet50_Weights

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
data_dir = './archive/Training'  # Path to your training data

# Load default weights and preprocessing for ResNet50
weights = ResNet50_Weights.DEFAULT

# Stronger data augmentations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    weights.transforms()  # Preprocessing from ResNet50 weights
])

# Load dataset
train_dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Print dataset info
print(f"Number of training images: {len(train_dataset)}")
print(f"Class names: {train_dataset.classes}")

# Define the model and unfreeze last few layers
model = models.resnet50(weights=weights)
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Modify the final fully connected layer
num_classes = len(train_dataset.classes)
model.fc = nn.Sequential(
    nn.Dropout(0.5),  # Dropout for regularization
    nn.Linear(model.fc.in_features, num_classes)
)

model = model.to(device)

#  Class weights for imbalanced classes
labels_tensor = torch.tensor([label for _, label in train_dataset.samples])
class_sample_count = torch.tensor([(labels_tensor == i).sum().item() for i in range(num_classes)])
class_weights = 1. / class_sample_count.float()
print("Class weights:", class_weights)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# Learning rate scheduler (reduce LR every 5 epochs)
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

#  Optional: Test one batch
sample_images, sample_labels = next(iter(train_loader))
print(f"Sample batch shape: {sample_images.shape}, Labels: {sample_labels}")

# Training
epochs = 25  # More epochs for better training
model.train()
print("Starting training loop...")

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs} started.")
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"  Processing Batch {batch_idx + 1}/{len(train_loader)}")

        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Step the scheduler
    scheduler.step()

    print(f"Epoch [{epoch + 1}/{epochs}] complete. Loss: {running_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), 'brain_tumor_resnet50_final.pth')
print(" Model training complete and saved as 'brain_tumor_resnet50_final.pth'")
