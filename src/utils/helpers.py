import os
import json
import subprocess

METRICS_PATH = "results/metrics.json"

# ------------------------------#
#         Metrics save          #
# ------------------------------#
def save_metrics(data):
    """Save results/metrics.json"""
    os.makedirs("results", exist_ok=True)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    else:
        metrics = []
    metrics.append(data)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved in {METRICS_PATH}")

# ------------------------------#
#      Hardware Detection       #
# ------------------------------#
def detect_vulkan():
    """We should detect if Vulkam is available first"""
    try:
        result = subprocess.run(["vulkaninfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if "Vulkan Instance Version" in result.stdout:
            return True
    except FileNotFoundError:
        pass
    return False