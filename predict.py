import os, json
import torch, timm
from PIL import Image
from torchvision import transforms
from pathlib import Path

#Change this for any of your test image path
IMG_PATH   = r"D:\path\to\any_image.jpg"

MODEL_NAME = "mobilevit_s"
OUT_DIR    = "MVIT"
CKPT_PATH  = str(Path(OUT_DIR) / "checkpoints" / "best.pth")
CLASSES_TXT = str(Path(OUT_DIR) / "artifacts" / "classes.txt")
IMG_SIZE   = 224
TOPK       = 5

def load_classes(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"classes file not found: {txt_path}")
    with open(txt_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = load_classes(CLASSES_TXT)
    num_classes = len(classes)

    # model
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    state = torch.load(CKPT_PATH, map_location="cpu")
    state = state.get("model", state)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

    #image prep
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5071,0.4865,0.4409),(0.2673,0.2564,0.2762)), ])

    img = Image.open(IMG_PATH).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topv, topi = probs.topk(TOPK)

    print("\nTop predictions:")
    for p, idx in zip(topv.tolist(), topi.tolist()):
        name = classes[idx] if idx < len(classes) else f"class_{idx}"
        print(f"  {name:20s}  prob={p:.4f}")

if __name__ == "__main__":
    main()
