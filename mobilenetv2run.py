import sys
import torch
from torchvision import models, transforms
from PIL import Image

#a device to select if no GPU
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Name of the device we used:", device)


CLASS_NAMES = ['bird', 'cat', 'dog', 'horse', 'ship'] #class names in order as training is done

#the model structure used in training
#starting from ImageNet-pretrained MobileNetV2, then replacing the last classifier layer to output 5 classes
#and loading the trained weights file from the MobileNetV2 with pretrained weights
base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

final_in_features = base_model.classifier[1].in_features #the no. of input features the classifier's final layer expects
new_output_layer = torch.nn.Linear(final_in_features, len(CLASS_NAMES)) #output for each 5 classes
base_model.classifier[1] = new_output_layer #new layer into the model structure after classifying

#Using the saved weights 
weights_path = "D:\\%%SLU\\Sem 3\\AI-Capstone\\Mobilenetv2\\mobilenetv2_tiny.pth"
state_dict = torch.load(weights_path, map_location=device)
base_model.load_state_dict(state_dict)
base_model = base_model.to(device)
base_model.eval()

#the same preprocessing pipeline as training- Resizing, Convert to tensor, Normalize
resize_step = transforms.Resize((224, 224))
to_tensor_step = transforms.ToTensor()
normalize_step = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225] )

preprocess = transforms.Compose([
    resize_step,
    to_tensor_step,
    normalize_step ])

#Checking on the image
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = "D:\\%%SLU\\Sem 3\\AI-Capstone\\Mobilenetv2\\cat.jpg"
print("Image path:", image_path)

try:
    pil_image = Image.open(image_path)
except Exception as e:
    print("Could not open image. Error:", str(e))
    sys.exit(1)

rgb_image = pil_image.convert("RGB")

image_tensor = preprocess(rgb_image)          #shape of: [3, 224, 224]
image_tensor = image_tensor.unsqueeze(0)      #shape of: [1, 3, 224, 224]
image_tensor = image_tensor.to(device)

#running the model to get predictions,Used probabilities with Softmax, Find highest prob
with torch.no_grad():
    logits = base_model(image_tensor)         # now shape: [1, 5]
    all_probs = torch.softmax(logits, dim=1)

probs_for_image = all_probs[0]
top_index_tensor = torch.argmax(probs_for_image)
top_index = int(top_index_tensor.item())

print("\nClass probabilities:")
i = 0
while i < len(CLASS_NAMES):
    class_name = CLASS_NAMES[i]
    prob_value = probs_for_image[i].item()
    percent_value = prob_value * 100.0
    print("{:<10s}: {:.2f}%".format(class_name, percent_value))
    i = i + 1

predicted_class_name = CLASS_NAMES[top_index]
print("\nPredicted class:", predicted_class_name)
