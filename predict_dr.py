import timm
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dr_model(model_path):
    model = timm.create_model('efficientnet_b0', pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 5)  # 5 DR classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_dr(image_path):
    model_path = "models/dr_effnetb0_fold2.pt"
    model = load_dr_model(model_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
    
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
    label = class_names[predicted_class.item()]
    conf = confidence.item() * 100

    return f"{label} ({conf:.2f}%)"
