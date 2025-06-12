"""Training script for NEU-DET with YOLOv8 + Swin Transformer.

This script loads the YOLOv8 model configuration that integrates Swin Transformer
layers and trains it on the NEU-DET dataset.  Paths are resolved relative to the
location of this file so that it works regardless of where the repository is
checked out.  It also selects the CPU automatically if a CUDA device is not
available to avoid `Invalid CUDA 'device=0'` errors.
"""

from pathlib import Path
import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent

# Choose the Swin Transformer backbone (one-layer structure by default)
MODEL_CFG = ROOT / "ultralytics" / "cfg" / "models" / "v8" / "yolov8_one_swinTrans.yaml"

# Load the model configuration
model = YOLO(str(MODEL_CFG))


if __name__ == "__main__":
    DATA_PATH = ROOT / "data" / "data.yaml"
    PRETRAINED = ROOT / "yolov8n.pt"

    device = "0" if torch.cuda.is_available() else "cpu"

    model.train(
        data=str(DATA_PATH),
        pretrained=str(PRETRAINED),
        epochs=400,
        imgsz=640,
        device=device,
    )

