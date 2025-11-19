import torch, torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image

CLASSES = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

def preprocess(pil):
    t = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),
                             (0.229,0.224,0.225)),
    ])
    return t(pil).unsqueeze(0)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = mobilenet_v3_small(num_classes=10).to(device)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
model.load_state_dict(torch.load("checkpoints/mnv3_cifar10_best.pt", map_location=device))
model.eval()

# test image
img = Image.open(r"D:\%%SLU\Sem 3\AI-Capstone\explainable-edge\app\cat.jpg").convert("RGB")  # put your own image path here
x = preprocess(img).to(device)

with torch.no_grad():
    logits = model(x)
    probs = torch.softmax(logits, 1)[0]

pred_idx = int(probs.argmax().item())
print(f"Predicted: {CLASSES[pred_idx]} ({probs[pred_idx]*100:.1f}%)")