import os, json
import torch, timm
from PIL import Image
from torchvision import transforms
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
#In IMG_PATH any pathof an Image can be setted to test its prediction But it should be in the same folder.
IMG_PATH = BASE_DIR / "test_images" / "car.jpg"
MODEL_NAME  = "mobilevit_s"
OUT_DIR     = "MVIT_C10"  #this should match with the location in train_c10.py
OUT_DIR_PATH = BASE_DIR / OUT_DIR
CKPT_PATH    = OUT_DIR_PATH / "checkpoints" / "best.pth"
CLASSES_TXT  = OUT_DIR_PATH / "artifacts" / "classes.txt"
IMG_SIZE    = 224
TOPK        = 5



def load_classes(txt_path: Path):
    if not txt_path.exists():
        raise FileNotFoundError(f"classes file not found: {txt_path}")
    with txt_path.open("r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    return classes


def main():
    if not IMG_PATH.exists():
        raise FileNotFoundError( f"Image not found: {IMG_PATH}. "
            "Place an image in this folder and update IMG_PATH.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    classes = load_classes(CLASSES_TXT)
    num_classes = len(classes)

#Loading the actual model here
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    state = torch.load(CKPT_PATH, map_location="cpu")
    state = state.get("model", state)
    model.load_state_dict(state, strict=False)
    model.eval().to(device)

#preperation of the image 
    tf = transforms.Compose([ transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),])

    img = Image.open(IMG_PATH).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        topv, topi = probs.topk(TOPK)

    print(f"\nTop predictions for: {IMG_PATH}")
    for p, idx in zip(topv.tolist(), topi.tolist()):
        name = classes[idx] if idx < len(classes) else f"class_{idx}"
        print(f"  {name:20s}  prob={p:.4f}")

if __name__ == "__main__":
    main()
