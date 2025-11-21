import sys, random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from PIL import Image

DATA_DIR   = r"D:\MVIT\data"
OUT_DIR    = r"D:\MVIT\MVIT_C10"
MODEL_NAME = "mobilevit_s"
IMG_SIZE   = 224

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

#explainability feature

def detect_cam_region(cam):
    cam = cam.cpu().numpy()
    h, w = cam.shape
    y, x = np.unravel_index(cam.argmax(), cam.shape)
    row = ["upper", "center", "lower"][y // (h//3)]
    col = ["left", "middle", "right"][x // (w//3)]
    return f"{row}-{col}"

def detect_edge_strength(img, cam):
    img_np = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edges = np.abs(sobelx) + np.abs(sobely)
    cam_resized = cv2.resize(cam.cpu().numpy(), (edges.shape[1], edges.shape[0]))
    score = (edges * cam_resized).mean()
    if score > 80: return "strong edges (clear object boundaries)"
    elif score > 40: return "moderate edges (partial outline)"
    else: return "weak edges (shape not very important)"

def detect_color_feature(img, cam):
    img_np = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    cam_np = cam.cpu().numpy()
    cam_resized = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]))
    mask = cam_resized > cam_resized.mean()
    masked_pixels = img_np[mask]
    if len(masked_pixels) == 0: return "color not significant"
    mean_color = masked_pixels.mean(axis=0)
    r, g, b = mean_color
    if r>g and r>b: return "predominantly red-toned region"
    if g>r and g>b: return "predominantly green-toned region"
    if b>r and b>g: return "predominantly blue-toned region"
    return "mixed color region"

def generate_model_explanation(classname, region, edge_info, color_info):
    return (
        f"The model predicted **{classname}** because its strongest attention was in "
        f"the **{region} region**.\n\n"
        f"It relied on **{edge_info}**, suggesting a focus on structural shape cues.\n\n"
        f"Additionally, the region included a **{color_info}**, indicating color also contributed.\n\n"
        f"Together, these cues influenced the model’s decision."
    )

#Interepretability Stability Score

def compute_iss(cam1, cam2):
    cam1 = cam1.astype(np.float32)
    cam2 = cam2.astype(np.float32)
    cam1 = (cam1 - cam1.min())/(cam1.max()-cam1.min()+1e-6)
    cam2 = (cam2 - cam2.min())/(cam2.max()-cam2.min()+1e-6)
    score,_ = ssim(cam1, cam2, full=True, data_range=1.0)
    return score

def apply_small_perturbation(img_tensor):
    pil = transforms.ToPILImage()(img_tensor)
    pil = TF.rotate(pil, random.uniform(-3,3))
    pil = TF.adjust_brightness(pil, random.uniform(0.9,1.1))
    t = transforms.ToTensor()(pil)
    t = (t + torch.randn_like(t)*0.01).clamp(0,1)
    for c in range(3): t[c] = (t[c] - MEAN[c]) / STD[c]
    return t

#Grad-CAM

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target = target_layer
        self.activations = None
        self.gradients = None
        self.target.register_forward_hook(self._save_activation)
        self.target.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, m,i,o): self.activations = o.detach()
    def _save_gradient(self, m,gin,gout): self.gradients = gout[0].detach()

    def __call__(self, logits, class_idx):
        self.model.zero_grad(set_to_none=True)
        one_hot = torch.zeros_like(logits)
        one_hot[0,class_idx] = 1.0
        logits.backward(gradient=one_hot)
        acts = self.activations
        grads = self.gradients
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= cam.min()
        cam /= (cam.max()+1e-6)
        return cam

#single image Demo
def run_single_image(model, gc, img_path, class_names, device):

    tf_viz = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor()
    ])

    tf_model = transforms.Compose([
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN,STD)
    ])

    #load image safely
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_pil = Image.fromarray(img_rgb)

    img_viz = tf_viz(img_pil)
    tensor = tf_model(img_pil).unsqueeze(0).to(device)

    logits = model(tensor)
    probs  = torch.softmax(logits,1)[0]
    pred   = int(probs.argmax())

    cam = gc(logits,pred)[0,0].cpu()

    perturbed = apply_small_perturbation(img_viz)
    logits2 = model(perturbed.unsqueeze(0).to(device))
    cam2 = gc(logits2,pred)[0,0].cpu()

    iss = compute_iss(cam.numpy(), cam2.numpy())

    region = detect_cam_region(cam)
    edge_info = detect_edge_strength(img_viz, cam)
    color_info = detect_color_feature(img_viz, cam)

    explanation = generate_model_explanation(
        class_names[pred], region, edge_info, color_info
    )

    disp = cv2.resize(img_rgb, (IMG_SIZE,IMG_SIZE))

    plt.figure(figsize=(12,5))
    plt.subplot(1,3,1); plt.imshow(disp); plt.title("Original"); plt.axis("off")
    plt.subplot(1,3,2); plt.imshow(disp, alpha=0.5); plt.imshow(cam, cmap="jet", alpha=0.5); plt.title("Grad-CAM"); plt.axis("off")
    plt.subplot(1,3,3); plt.text(0,0.5,f"Pred: {class_names[pred]}\nISS={iss:.3f}\n\n{explanation}", fontsize=10, wrap=True); plt.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    out_dir = Path(OUT_DIR)
    class_file = out_dir/"artifacts"/"classes.txt"
    ckpt_file  = out_dir/"checkpoints"/"best.pth"

    class_names = class_file.read_text().splitlines()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(class_names)).to(device)
    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()

    target = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            target = m

    gc = GradCAM(model, target)

    if len(sys.argv) > 1:
        run_single_image(model, gc, sys.argv[1], class_names, device)
        return

    print("Usage: python demo_gradcam.py path/to/image.jpg")

if __name__ == "__main__":
    main()
    