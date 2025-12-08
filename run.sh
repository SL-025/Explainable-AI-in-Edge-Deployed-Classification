#!/usr/bin/env bash
set -e

if command -v cygpath >/dev/null 2>&1; then
  ROOT_DIR=$(cygpath -w "$(pwd)")
else
  ROOT_DIR=$(pwd)
fi

CMD="$1"
shift || true

case "$CMD" in
  build)
    echo ">>> Building Docker image explainable-vit..."
    docker build -t explainable-vit .
    ;;

  train)
    echo ">>> Training MobileViT-S inside Docker..."
    docker run --rm \
      -v "$ROOT_DIR/data:/app/data" \
      -v "$ROOT_DIR/src/MVIT_C10:/app/src/MVIT_C10" \
      explainable-vit \
      python src/train_c10.py
    ;;

  gradcam)
    echo ">>> Running CIFAR-10 Grad-CAM + ISS (full test set)..."
    docker run --rm \
      -v "$ROOT_DIR/data:/app/data" \
      -v "$ROOT_DIR/src/MVIT_C10:/app/src/MVIT_C10" \
      explainable-vit \
      python src/iss_gradcam.py
    echo ">>> Check: src/MVIT_C10/results/iss_gradcam_cifar10.png"
    ;;

  gradcam_one)
    echo ">>> Running ISS + Grad-CAM on one custom image..."
    docker run --rm \
      -v "$ROOT_DIR/src/MVIT_C10:/app/src/MVIT_C10" \
      -v "$ROOT_DIR/test_images:/app/test_images" \
      explainable-vit \
      python src/iss_gradcam_oneimage.py
    echo ">>> Check: src/MVIT_C10/OneImage/OneImage.png"
    ;;

  predict)
    echo ">>> Running probability prediction on one image..."
    docker run --rm \
      -v "$ROOT_DIR/src/MVIT_C10:/app/src/MVIT_C10" \
      -v "$ROOT_DIR/test_images:/app/test_images" \
      explainable-vit \
      python src/predict.py
    ;;

  *)
    echo "Usage: ./run.sh {build|train|gradcam|gradcam_one|predict}"
    exit 1
    ;;
esac
