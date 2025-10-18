"""
HVAC YOLOv11 Training Module (CLI Version)
------------------------------------------
Train and export YOLOv11 for HVAC equipment detection.

Usage examples:
---------------
    python src/train.py
    python src/train.py --epochs 150 --model yolo11m.pt --imgsz 512
    python src/train.py --export-formats onnx ncnn openvino
"""

import os
import json
import yaml
import logging
import argparse
import torch
from datetime import datetime
from dotenv import load_dotenv
from ultralytics import YOLO
from roboflow import Roboflow

# ----------------------------#
#    Logging configuration    #
# ----------------------------#
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("results/train_logs.txt", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

if not torch.cuda.is_available():
    logger.warning("CUDA not available. Training will use CPU — expect slower performance.")
else:
    logger.info("CUDA device detected: %s", torch.cuda.get_device_name(0))

# --------------------#
#  Dataset Download   #
# --------------------#
def download_dataset(api_key):
    logger.info("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("hvac-hbzel").project("hvac_8_class-lfhrn")
    version = project.version(4)
    dataset = version.download("yolov11")
    logger.info("Dataset downloaded to: %s", dataset.location)
    return dataset.location

# ------------------------#
#   Training Function     #
# ------------------------#
def train_model(data_yaml, model_name="yolo11s.pt", epochs=100, imgsz=640, batch=16, device="cpu"):
    logger.info("Starting YOLOv11 training...")
    model = YOLO(model_name)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project="runs/train",
        name="hvac_yolov11",
        plots=True
    )
    logger.info("Training complete. Best model saved at: runs/train/hvac_yolov11/weights/best.pt")
    return results

# -------------------------#
#      Export Function     #
# -------------------------#
def export_models(model_path, imgsz=640, formats=("ncnn", "openvino", "edgetpu")):
    model = YOLO(model_path)
    exported = []
    for fmt in formats:
        try:
            model.export(format=fmt, imgsz=imgsz)
            exported.append(fmt)
            logger.info(f"Exported format: {fmt}")
        except Exception as e:
            logger.warning(f"Failed to export {fmt}: {e}")
    return exported

# ------------------------------#
#    Metrics Save Function      #
# ------------------------------#
def save_metrics(results):
    metrics = {
        "datetime": datetime.now().isoformat(),
        "mAP50": results.results_dict.get("metrics/mAP50(B)", None),
        "precision": results.results_dict.get("metrics/precision(B)", None),
        "recall": results.results_dict.get("metrics/recall(B)", None),
        "epochs": results.epochs,
        "train_loss": results.results_dict.get("train/box_loss", None),
    }

    os.makedirs("results", exist_ok=True)
    with open("results/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("Metrics saved → results/train_metrics.json")

# ------------------------------------------#
#       Main Function with CLI Arguments    #
# ------------------------------------------#
def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 HVAC detection model.")
    parser.add_argument("--model", default="yolo11s.pt", help="Base model to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu, cuda, 0, etc.)")
    parser.add_argument("--export-formats", nargs="+", default=["ncnn", "openvino", "edgetpu"], help="List of export formats")
    parser.add_argument("--skip-download", action="store_true", help="Skip dataset download if already available")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    rf_key = os.getenv("ROBOFLOW_KEY")
    if not rf_key:
        raise EnvironmentError("Missing ROBOFLOW_KEY. Define it in .env or as environment variable.")

    dataset_path = "datasets/HVAC_8_CLASS-4"
    if not args.skip_download or not os.path.exists(dataset_path):
        dataset_path = download_dataset(rf_key)

    data_yaml = os.path.join(dataset_path, "data.yaml")
    results = train_model(
        data_yaml=data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device
    )

    try:
        names = results.model.names if hasattr(results.model, "names") else {}
        class_metrics = {}

        # Extract class-level metrics if YOLO provides them
        if hasattr(results, "metrics") and hasattr(results.metrics, "class_result"):
            for i, cls_name in names.items():
                if i < len(results.metrics.class_result):
                    vals = results.metrics.class_result[i]
                    class_metrics[cls_name] = {
                        "precision": float(vals[0]),
                        "recall": float(vals[1]),
                        "mAP50": float(vals[2])
                    }

        os.makedirs("results", exist_ok=True)
        with open("results/per_class_metrics.json", "w") as f:
            json.dump(class_metrics, f, indent=4)
        logger.info("Per-class metrics saved → results/per_class_metrics.json")

    except Exception as e:
        logger.warning(f"Could not extract per-class metrics: {e}")

    save_metrics(results)

    best_model = "runs/train/hvac_yolov11/weights/best.pt"
    export_models(best_model, imgsz=args.imgsz, formats=args.export_formats)

    logger.info("Training and export completed successfully.")

if __name__ == "__main__":
    main()