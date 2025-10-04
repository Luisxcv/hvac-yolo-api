import cv2
import time
import os
import json
import torch
import sys
import subprocess
from ultralytics import YOLO
from src.utils.helpers import save_metrics
from src.utils.visualization import draw_boxes
from src.utils.backends import InferenceBackend, TorchBackend, OpenVINOBackend, NCNNBackend
from src.utils.core import run_file, run_camera

# ------------------------------#
#         Model paths           #
# ------------------------------#
MODEL_PT = "models/final/best.pt"                    # PyTorch (Nvidia CUDA)
MODEL_OPENVINO = "models/final/best_openvino_model"  # OpenVINO (Intel IrisXe)
MODEL_NCNN = "models/final/best_ncnn_model/"               # NCNN (low end optimized)

# OUTPUT paths
OUTPUT_DIR = "results/sample_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
METRICS_PATH = "results/metrics.json"

# ------------------------------#
#    Automatic Model Selection  #
# ------------------------------#
def select_backend():
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU)")
        return TorchBackend(MODEL_PT)
    elif os.path.exists(MODEL_OPENVINO):
        print("Using OpenVINO (Intel CPU/GPU)")
        return OpenVINOBackend(MODEL_OPENVINO)
    else:
        print("Using NCNN")
        return NCNNBackend(MODEL_NCNN)

# ---------------------------#
#           Main             #
# ---------------------------#
if __name__ == "__main__":
    backend = select_backend()

    print("Choose inference mode:")
    print("1. Camera")
    print("2. Local file (image or video)")
    choice = input("👉 Type 1 or 2: ")

    if choice == "1":
        run_camera(backend)
    elif choice == "2":
        run_file(backend)
    else:
        print("❌ Invalid option")