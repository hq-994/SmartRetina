import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_gl_model(model_path):
    model = timm.create_model('efficientnet_b0', pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier.in_features, 2)
    )
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_gl(image_path):
    model_path = 'models/best_model.pth'
    model = load_gl_model(model_path)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)

        # 🔍 DEBUG: Show actual class probabilities
        print("🧠 Glaucoma Probabilities:", probs.cpu().numpy())

    class_names = ['No Glaucoma', 'Glaucoma']
    label = class_names[predicted_class.item()]
    conf = confidence.item() * 100

    return f"{label} ({conf:.2f}%)"
