import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision import transforms
import timm

from iss_gradcam import (GradCAM, compute_iss, apply_small_perturbation,
    detect_cam_region, detect_edge_strength, detect_color_feature,)

BASE_DIR = Path(__file__).resolve().parent
IMG_PATH = BASE_DIR / "test_images" / "cat.jpg"
#Update the image Name to test in IMG_PATH
OUT_DIR    = os.environ.get("MVIT_OUT_DIR", "MVIT_C10")
MODEL_NAME = "mobilevit_s"
IMG_SIZE   = 224

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)

def find_last_conv(m: torch.nn.Module):
    last = None
    for mod in m.modules():
        if isinstance(mod, torch.nn.Conv2d):
            last = mod
    return last

def load_img(path):
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),transforms.Normalize(MEAN, STD),])
    return t(img)

def unnorm(x):
    x = x.clone()
    for c in range(3):
        x[c] = x[c] * STD[c] + MEAN[c]
    return x.clamp(0, 1)

def run():
    if not IMG_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMG_PATH}. Place an image in this folder and "
            "update IMG_PATH in iss_gradcam_oneimage.py")

    print("\nImage:", IMG_PATH)

    classes_path = Path(OUT_DIR) / "artifacts" / "classes.txt"
    ckpt_path    = Path(OUT_DIR) / "checkpoints" / "best.pth"

    if not classes_path.exists():
        raise FileNotFoundError(f"classes.txt not found at {classes_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best.pth not found at {ckpt_path}")

    classes = classes_path.read_text(encoding="utf-8").splitlines()
    ckpt = torch.load(ckpt_path, map_location="cpu")

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=len(classes))
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Device:", device)

    target_layer = find_last_conv(model)
    if target_layer is None:
        try:
            target_layer = model.patch_embed.proj
            print("Using patch_embed.proj as target layer.")
        except Exception:
            raise RuntimeError("No Conv2d layer found for Grad-CAM.")

    gc = GradCAM(model, target_layer)
    x_norm = load_img(IMG_PATH)
    x = x_norm.unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(probs.argmax())
    pred_prob = float(probs[pred_idx])

    cam = gc(logits, class_idx=pred_idx)
    cam_up = F.interpolate(cam, size=x_norm.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    cam_up = cam_up.clamp(0, 1).cpu()

    pert = apply_small_perturbation(x_norm)
    x2 = pert.unsqueeze(0).to(device)
    logits2 = model(x2)
    cam2 = gc(logits2, class_idx=pred_idx)
    cam2_up = F.interpolate(cam2, size=x_norm.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    cam2_up = cam2_up.clamp(0, 1).cpu()

    iss_score = compute_iss(cam_up.numpy(), cam2_up.numpy())
    region = detect_cam_region(cam_up)
    edge_info = detect_edge_strength(x_norm, cam_up)
    color_info = detect_color_feature(x_norm, cam_up)

    explanation = (f"Prediction: {classes[pred_idx]}\n"
        f"ISS Score: {iss_score:.3f}\n\n"
        f"The model predicted **{classes[pred_idx]}** because most activation "
        f"concentrated in the **{region}** region.\n"
        f"It primarily focused on the object's **{edge_info}**, which indicates the "
        f"model relied on shape and structural boundaries.\n"
        f"The highlighted area also showed a **{color_info}**, suggesting color "
        f"played a supporting role in the decision.\n"
        "Together, these visual cues guided the model toward its final prediction.")

    disp = unnorm(x_norm).permute(1, 2, 0).cpu().numpy()
    fig = plt.figure(figsize=(18, 4))
    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 2.2, 0.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(disp)
    ax1.axis("off")
    ax1.set_title(f"Original\nPred: {classes[pred_idx]} ({pred_prob:.2f})",
        fontsize=10,)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(disp, alpha=0.6)
    ax2.imshow(cam_up, cmap="jet", alpha=0.45)
    ax2.axis("off")
    ax2.set_title("Grad-CAM", fontsize=10)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.set_title("Explanation", fontsize=12)
    ax3.text(0, 1, explanation, fontsize=9,
        ha="left", va="top", wrap=True,)

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis("off")
    plt.tight_layout()
    plt.show()
    gc.close()
    print("\nDone.")

if __name__ == "__main__":
    run()
