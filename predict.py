import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# Load the trained model
model_path = "brain_tumor_resnet50.pth"  # Update if the filename is different
model = models.resnet50()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 4)  
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class names (order should match training classes)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

def predict_image(image_path):
    if not os.path.exists(image_path):
        print("Image not found! Check the path.")
        return
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_label = class_names[predicted.item()]
    
    print(f"Prediction: {class_label}")
    

image_path = r"C:\Users\misht\OneDrive\Desktop\MRI\archive\Testing\glioma\Te-gl_0010.jpg"
predict_image(image_path)
