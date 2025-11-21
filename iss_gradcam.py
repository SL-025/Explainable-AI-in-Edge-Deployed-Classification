import sys, random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2



DATA_DIR   = r"D:\MVIT\data"
OUT_DIR    = r"D:\MVIT\MVIT_C10"
MODEL_NAME = "mobilevit_s" 
IMG_SIZE   = 224
N_SAMPLES  = 6

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

#explainibility
def explain_prediction(class_name, cam):
    """
    Generate a simple human-friendly explanation using CAM heatmap.
    """

    cam_np = cam.cpu().numpy()
    h, w = cam_np.shape

    top = cam_np[:h//3, :].mean()
    mid = cam_np[h//3:2*h//3, :].mean()
    bottom = cam_np[2*h//3:, :].mean()

    if mid == max(top, mid, bottom):
        region = "central region"
    elif top == max(top, mid, bottom):
        region = "upper region"
    else:
        region = "lower region"

    #simple CIFAR 10 class based templates
    templates = {
        "airplane": "The model focused on the wings and body structure typical of airplanes.",
        "automobile": "The model highlighted the wheels and chassis shape found in cars.",
        "bird": "The model concentrated on the head and wing region typical of birds.",
        "cat": "The model focused on the head and fur texture distinctive for cats.",
        "deer": "The model concentrated on the face and upper body of the deer.",
        "dog": "The model emphasized the head and torso features typical of dogs.",
        "frog": "The model focused on the body texture and shape typical of frogs.",
        "horse": "The model focused on the long head and body characteristic of horses.",
        "ship": "The model detected the hull and body outline characteristic of ships.",
        "truck": "The model focused on the rectangular cabin and wheels typical of trucks."
    }

    base = templates.get(class_name, "The model focused on key visual features.")
    return f"{base} Most activation was in the {region} of the image."

def detect_cam_region(cam):
    """
    Detects the region of highest activation in the CAM.
    Returns: string like 'upper-left', 'center', 'lower-right'
    """
    cam = cam.cpu().numpy()
    h, w = cam.shape

    y, x = np.unravel_index(cam.argmax(), cam.shape)

    row = ["upper", "center", "lower"][y // (h//3)]
    col = ["left", "middle", "right"][x // (w//3)]

    return f"{row}-{col}"

def detect_edge_strength(img, cam):
    """
    Detect how strong edges overlap with CAM area.
    Returns: 'strong edges', 'moderate edges', or 'weak edges'
    """
    img_np = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    #Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    edges = np.abs(sobelx) + np.abs(sobely)

    #resize CAM to match edge map
    cam_resized = cv2.resize(cam.cpu().numpy(), (edges.shape[1], edges.shape[0]))

    #overlap score
    score = (edges * cam_resized).mean()

    if score > 80:
        return "strong edges (clear object boundaries)"
    elif score > 40:
        return "moderate edges (partial outline)"
    else:
        return "weak edges (shape not very important)"

def detect_color_feature(img, cam):
    """
    Determine if color is part of the model's reasoning.
    """
    img_np = (img.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    cam_np = cam.cpu().numpy()

    #resize CAM to match image
    cam_resized = cv2.resize(cam_np, (img_np.shape[1], img_np.shape[0]))

    #mask region where CAM is highest
    mask = cam_resized > cam_resized.mean()
    masked_pixels = img_np[mask]

    if len(masked_pixels) == 0:
        return "color not significant"

    #dominant color
    mean_color = masked_pixels.mean(axis=0)
    r, g, b = mean_color
    
    #simple color category
    if r > g and r > b:
        return "predominantly red-toned region"
    elif g > r and g > b:
        return "predominantly green-toned region"
    elif b > r and b > g:
        return "predominantly blue-toned region"
    else:
        return "mixed color region"

def generate_model_based_explanation(class_name, region, edge_info, color_info):
    explanation = (
        f"The model predicted **{class_name}** because its attention was strongest "
        f"in the **{region} region** of the image.\n\n"
        f"The Grad-CAM map shows the model relied on **{edge_info}**, indicating it "
        f"was focusing on the object's structure or outline.\n\n"
        f"In addition, the model attended to a **{color_info}**, suggesting color "
        f"also contributed to the prediction.\n\n"
        f"These combined visual cues influenced the model's final decision."
    )
    return explanation


def compute_iss(cam1, cam2):
    """
    Compute Interpretability Stability Score (ISS).
    cam1, cam2 are numpy arrays in [0,1].
    Using SSIM (structural similarity) as similarity measure.
    """
    cam1 = cam1.astype(np.float32)
    cam2 = cam2.astype(np.float32)

    #normalize
    cam1 = (cam1 - cam1.min()) / (cam1.max() - cam1.min() + 1e-6)
    cam2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min() + 1e-6)

    score, _ = ssim(cam1, cam2, full=True, data_range=1.0)
    return score

import torchvision.transforms.functional as TF

def apply_small_perturbation(img_tensor):
    """
    Apply a tiny random perturbation to test stability:
    - slight rotation (±3°)
    - slight brightness change
    - tiny gaussian noise
    """
    pil = transforms.ToPILImage()(img_tensor)

    #random small rotation
    angle = random.uniform(-3, 3)
    pil = TF.rotate(pil, angle)

    pil = TF.adjust_brightness(pil, random.uniform(0.9, 1.1))

    t = transforms.ToTensor()(pil)

    noise = torch.randn_like(t) * 0.01
    t = (t + noise).clamp(0, 1)

    #normalize again
    for c in range(3):
        t[c] = (t[c] - MEAN[c]) / STD[c]

    return t


#Gradcam
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._save_activation)
        self._bh = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, logits, class_idx=None):
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad(set_to_none=True)
        one_hot = torch.zeros_like(logits)
        one_hot[0, class_idx] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)

        acts = self.activations
        grads = self.gradients
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def close(self):
        self._fh.remove()
        self._bh.remove()


def find_last_conv(m: nn.Module):
    last = None
    for mod in m.modules():
        if isinstance(mod, nn.Conv2d):
            last = mod
    return last

def main():
    print("== CIFAR-10 Grad-CAM (MobileViT) ==")
    print("OUT_DIR:", OUT_DIR)
    print("DATA_DIR:", DATA_DIR)

    out_dir = Path(OUT_DIR)
    classes_path = out_dir / "artifacts" / "classes.txt"
    ckpt_path    = out_dir / "checkpoints" / "best.pth"

    if not classes_path.exists():
        print("ERROR: classes.txt not found.")

        sys.exit(1)
    if not ckpt_path.exists():
        print("ERROR: best.pth not found.")
        sys.exit(1)

    class_names = [ln.strip() for ln in classes_path.read_text(encoding="utf-8").splitlines()]
    num_classes = len(class_names)
    print(f"Loaded {num_classes} class names.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    #model
    print(f"Building model: {MODEL_NAME}")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    print("Model weights loaded.")

    #find target conv layer
    target_conv = find_last_conv(model)
    if target_conv is None:
        try:
            target_conv = model.patch_embed.proj
            print("INFO: Using patch_embed.proj as target conv (ViT).")
        except Exception:
            print("ERROR: No Conv2d layer found for Grad-CAM.")
            sys.exit(1)
    else:
        print("Using target conv layer:", target_conv.__class__.__name__)

    gc = GradCAM(model, target_conv)

    #dataset
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    print("Preparing CIFAR-10 test set...")
    test_ds = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=tf)
    print("Test images:", len(test_ds))

    # visualization
    N = min(N_SAMPLES, len(test_ds))
    print(f"Visualizing {N} random images...")
    plt.figure(figsize=(10, 2*N+2))

    for i in range(N):
        idx = random.randrange(0, len(test_ds))
        img, true_id = test_ds[idx]
        x = img.unsqueeze(0).to(device)

        #forward pass WITH gradients
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_id = int(probs.argmax().item())
        pred_p  = float(probs[pred_id].item())

        cam = gc(logits, class_idx=pred_id)
        cam_up = F.interpolate(cam, size=img.shape[1:], mode='bilinear', align_corners=False)[0,0]
        cam_up = cam_up.clamp(0,1).cpu()

        #create perturbed version of the image
        perturbed_img = apply_small_perturbation(img)
        x2 = perturbed_img.unsqueeze(0).to(device)

        logits2 = model(x2)
        cam2 = gc(logits2, class_idx=pred_id)
        cam2_up = F.interpolate(cam2, size=img.shape[1:], mode='bilinear', align_corners=False)[0,0]
        cam2_up = cam2_up.clamp(0,1).cpu()

        #compute ISS
        iss_score = compute_iss(cam_up.numpy(), cam2_up.numpy())
        print(f"ISS (stability) for {class_names[pred_id]}: {iss_score:.3f}")

        region = detect_cam_region(cam_up)
        edge_info = detect_edge_strength(img, cam_up)
        color_info = detect_color_feature(img, cam_up)

        explanation = generate_model_based_explanation(
            class_names[pred_id],
            region,
            edge_info,
            color_info
        )
        print(f"[{class_names[pred_id]}] -> {explanation}")


        #denormalize image
        disp = img.clone()
        for c in range(3):
            disp[c] = disp[c]*STD[c] + MEAN[c]
        disp = disp.clamp(0,1).permute(1,2,0).cpu().numpy()

        plt.subplot(N, 2, 2*i+1)
        plt.imshow(disp); plt.axis('off')
        plt.title(
            f"GT: {class_names[true_id]}\nPred: {class_names[pred_id]} ({pred_p:.2f})",
            fontsize=9
        )

        #heatmap + explanation
        plt.subplot(N, 2, 2*i+2)
        plt.imshow(disp, alpha=0.6)
        plt.imshow(cam_up, cmap='jet', alpha=0.4)
        plt.axis('off')
        plt.title(f"Grad-CAM\n{explanation}\nISS={iss_score:.2f}", fontsize=7)

    plt.tight_layout()
    plt.show()
    gc.close()
    print("Done.")


if __name__ == "__main__":
    main()
