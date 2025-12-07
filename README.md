
# Explainable AI in Edge-Deployed Classification

This project implements an **Explainable Image Classification Pipeline** using the **MobileViT-S** model trained on CIFAR‑10, with full support for **Grad‑CAM**, **Interpretability Stability Score (ISS)**, and **Docker-based reproducibility** for any clean machine.

The repository is structured to match the project requirements:

```
/src
    /train_c10.py
    /iss_gradcam.py
    /iss_gradcam_oneimage.py 
    /predict.py
    /MVIT_C10
        /artifacts
            /classes.txt
            /config.json
            /metrics.csv
        /checkpoints
            /best.pth
            /last.pth
        /tb
            /tensorboard events file
/docs
    /experiments
/poster
    /poster.pdf
/report
    /FinalReport.pdf
    /LatexCode
/reports
    /Weeklyreport.pdf
run.sh
requirements.txt
Dockerfile
```

Additional folders which will be created automatically at runtime of Training and iss_gradcam, If data is not available:
```
/data          → CIFAR-10 dataset (auto-downloaded)
/src/MVIT_C10      → checkpoints, artifacts, metrics, results
/test_images   → user images for prediction and one-image GradCAM
```

This README explains *exactly how to run the project*, both locally and using **Docker**.

# 1. PREREQUISITES
Anyone can run this project in **two ways**:

## OPTION A — Local Python (simple, but requires dependencies)

Requirements:
- Python 3.10+
- pip
- Git or ZIP download

## OPTION B — Docker (recommended, for on any system)

Requirements:
- Docker Desktop (Windows/macOS) OR Docker Engine (Linux)
Check installation: docker --version

# 2. LOCAL PYTHON USAGE (OPTION-A)
## 2.1. Download the Repository

### ZIP:
Download → Extract → Open terminal inside the folder.

### Git:
git clone https://github.com/SL-025/Explainable-AI-in-Edge-Deployed-Classification
cd Explainable-AI-in-Edge-Deployed-Classification

## 2.2. Create a Virtual Environment(this is optional)

### Windows PowerShell
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```
### Linux/macOS
```
python3 -m venv .venv
source .venv/bin/activate
```

## 2.3. Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```

## 2.4. Using Pretrained Checkpoints (Recommended)

To run prediction, Grad‑CAM and iss First check for the following files:
We already have this file
```
src/MVIT_C10/
  checkpoints/
     best.pth
  artifacts/
     classes.txt
     config.json
```
If these are already present then nothing more is needed.
If not present then first Run Training [2.5, train_c10.py]

## 2.5. Training (train_c10.py)
### This is very Important to run code in SRC folder, so run " cd src" before running scripts.
To retrain:
```
python train_c10.py
```
This will:

- Download CIFAR‑10 into `/data`
- Create/update `/src/MVIT_C10/checkpoints/*.pth`
- Log artifacts into `/src/MVIT_C10/artifacts/`

> !!! Retraining **overwrites best.pth**.  
> If you want to keep the original checkpoint, rename the folder:

```
mv src/MVIT_C10 src/MVIT_C10_backup
```

---

## 2.6. Prediction (predict.py)
### This is very Important to run code in SRC folder, so run " cd src" before running scripts.
1. Put an image inside:

```
test_images/car.jpg
```

2. Open `predict.py` and set:

```
IMG_PATH = "test_images/car.jpg"
```

3. Run:

```
cd src
python predict.py
```

---

## 2.7. Grad‑CAM + ISS on CIFAR‑10 Test Set (iss_gradcam.py)
### This is very Important to run code in SRC folder, so run " cd src" before running scripts.

```
cd src
python iss_gradcam.py
```

Output saved to:

```
src/MVIT_C10/results/iss_gradcam_cifar10.png
```

---

## 2.8. Grad‑CAM + ISS on a Single Custom Image (iss_gradcam_oneimage.py)
### This is very Important to run code in SRC folder, so run " cd src" before running scripts.

1. Add an image:

```
test_images/cat.jpg
```

2. Set path in script:

```
IMG_PATH = "test_images/cat.jpg"
```

3. Run:

```
cd src
python iss_gradcam_oneimage.py
```

Saved results appear under `src/MVIT_C10/OneImage/OneImage.png`.

---

# 3. DOCKER USAGE (RECOMMENDED)
Dockerfile:

```
FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y build-essential git libgl1 libglib2.0-0
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-c", "print('Container is ready...')"]
```

This creates a clean, reproducible environment.

---

## 3.1. Build Docker Image
### Windows PowerShell and macOS/Linux
### TO run this you should be in the folder: "Explainable-AI-in-Edge-Deployed-Classification"

```
docker build -t explainable-vit .
```
---

## 3.2. Train Inside Docker
### PowerShell
```
docker run --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/src/MVIT_C10:/app/src/MVIT_C10 `
  explainable-vit `
  python train_c10.py
```

### macOS/Linux
```
docker run --rm   -v "$(pwd)/data:/app/data"   -v "$(pwd)/src/MVIT_C10:/app/src/MVIT_C10"   explainable-vit   python train_c10.py
```

---

## 3.3. CIFAR‑10 Grad‑CAM + ISS (iss_gradcam.py)

PowerShell:

```
docker run --rm `
  -v ${PWD}/data:/app/data `
  -v ${PWD}/src/MVIT_C10:/app/src/MVIT_C10 `
  explainable-vit `
  python src/iss_gradcam.py
```

Creates:

```
src/MVIT_C10/results/iss_gradcam_cifar10.png
```

---

## 3.4. Custom Image Grad‑CAM + ISS (iss_gradcam_oneimage.py)

PowerShell:

```
docker run --rm `
  -v ${PWD}/src/MVIT_C10:/app/src/MVIT_C10 `
  -v ${PWD}/test_images:/app/test_images `
  explainable-vit `
  python src/iss_gradcam_oneimage.py
```

---

## 3.5. Prediction (predict.py)

PowerShell:

```
docker run --rm `
  -v ${PWD}/src/MVIT_C10:/app/src/MVIT_C10 `
  -v ${PWD}/test_images:/app/test_images `
  explainable-vit `
  python src/predict.py
```

---

# 4. run.sh (Automation Script)

A helper script is provided to simplify Docker usage:

```
./run.sh build
./run.sh train
./run.sh gradcam
./run.sh gradcam_one
./run.sh predict
```

---

# 5. DATASET HANDLING

- CIFAR‑10 automatically downloads into `/data`.
- No manual download needed.
- Changing dataset path:

```
export CIFAR10_DIR=/custom/path       (Linux/macOS)
setx CIFAR10_DIR "D:\custom\path"   (Windows)
```

---

# 6. TRAIN MORE / MODIFY MODEL

To train longer or modify hyperparameters:

1. Edit `train_c10.py` (epochs, LR, augmentations).
2. Backup existing outputs if needed.
3. Run the training command (local or Docker).

Explainability scripts always load:

```
src/MVIT_C10/checkpoints/best.pth
```

so new training results propagate automatically.

---

# 7. FUTURE WORK

- **Edge Deployment**  
  Export MobileViT-S to ONNX/TFLite for Android/iOS and micro‑edge boards.

- **Integrate Attention Rollout & Flow**  
  To visualize ViT internal attention propagation.

- **Token Diffusion & Attribution**  
  For deeper Transformer interpretability.

- **Compare with DINOv2**  
  Evaluate self‑supervised ViT backbones for stability + edge feasibility.

- **Real‑time Explainability Dashboard**  
  Web UI showing predictions + Grad‑CAM + ISS on device.

- **Robustness Evaluation**  
  Measure ISS stability under image perturbations (noise, flips, lighting).

---

# 8. SUMMARY FOR NEW USERS

1. Download the repo  
2. Use Python OR Docker  
3. If using pretrained checkpoint → ready to run explainability  
4. If training again → run `train_c10.py`  
5. Use Docker commands or `run.sh` for full reproducibility  
6. View results under:

```
src/MVIT_C10/checkpoints/
src/MVIT_C10/artifacts/
src/MVIT_C10/results/
src/MVIT_C10/OneImage
```

The project is fully reproducible on **any machine**, regardless of OS or Python version.
