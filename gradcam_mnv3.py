import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def _last_conv(model: nn.Module) -> nn.Module:
    """Return the last Conv2d layer in the model (good target for Grad-CAM)."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d found for Grad-CAM.")
    return last

def _prep_for_model(pil_img: Image.Image) -> torch.Tensor:
    """Preprocess for MobileNetV3 (224, normalize). Returns (1,3,224,224) tensor."""
    t = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return t(pil_img).unsqueeze(0)

def _loc_words(cam: np.ndarray) -> str:
    """Tiny heuristic to describe where heat is concentrated."""
    yy, xx = np.meshgrid(np.arange(cam.shape[0]), np.arange(cam.shape[1]), indexing='ij')
    s = cam.sum() + 1e-8
    cy = int((cam * yy).sum() / s)
    cx = int((cam * xx).sum() / s)
    ylab = ["top","upper","center","lower","bottom"][min(4, cy * 5 // cam.shape[0])]
    xlab = ["left","left-center","center","right-center","right"][min(4, cx * 5 // cam.shape[1])]
    return f"{ylab} {xlab}"

def explain(img_path: str,
            ckpt: str = "checkpoints/mnv3_cifar10_best.pt",
            out_path: str = "results/cam_overlay.png",
            device: str | None = None):
    # --- basic checks
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt} — train first or fix the path.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    pil = Image.open(img_path).convert("RGB")
    orig_w, orig_h = pil.size
    x = _prep_for_model(pil).to(device)

    model = mobilenet_v3_small(num_classes=10)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
    sd = torch.load(ckpt, map_location=device)
    model.load_state_dict(sd)
    model = model.to(device).eval()

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
    pred_idx = int(torch.argmax(probs).item())
    pred_cls = CLASSES[pred_idx]
    conf = float(probs[pred_idx].item()) * 100.0

    target_layer = _last_conv(model)
    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=x, targets=[ClassifierOutputTarget(pred_idx)])[0]  # (224,224)

    if _HAS_CV2:
        cam_resized = cv2.resize(grayscale_cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    else:
        cam_resized = np.array(Image.fromarray((grayscale_cam * 255).astype(np.uint8)).resize((orig_w, orig_h),
                                                                                              resample=Image.BILINEAR)) / 255.0

    rgb = np.array(pil).astype(np.float32) / 255.0  # (H,W,3)
    overlay = show_cam_on_image(rgb, cam_resized, use_rgb=True)

    if _HAS_CV2:
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    else:
        Image.fromarray(overlay).save(out_path)

    rationale = f"Predicted {pred_cls} ({conf:.1f}%). Evidence near {_loc_words(cam_resized)}."
    print(rationale)
    print(f"Saved overlay → {out_path}")
    return pred_cls, conf, out_path, rationale

def _build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img",  type=str, default=os.path.join("app", "cat.jpg"),
                    help="Path to an input image (jpg/png).")
    ap.add_argument("--ckpt", type=str, default=os.path.join("checkpoints", "mnv3_cifar10_best.pt"),
                    help="Path to model checkpoint (.pt).")
    ap.add_argument("--out",  type=str, default=os.path.join("results", "cat.png"),
                    help="Path to save overlay image.")
    return ap

if __name__ == "__main__":
    args = _build_argparser().parse_args()
    explain(img_path=args.img, ckpt=args.ckpt, out_path=args.out)
