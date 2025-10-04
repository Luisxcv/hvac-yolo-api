from ultralytics import YOLO
from src.utils.helpers import detect_vulkan

# -----------------------------#
#         Base Backend         #
# -----------------------------#
class InferenceBackend:
    def __init__(self, device):
        self.device = device
    def predict(self, frame):
        raise NotImplementedError

class TorchBackend(InferenceBackend):
    def __init__(self, model_path):
        super().__init__("CUDA")
        self.model = YOLO(model_path).to("cuda")
    def predict(self, frame):
        return self.model(frame, verbose=False)

class OpenVINOBackend(InferenceBackend):
    def __init__(self, model_dir):
        super().__init__("OpenVINO")
        self.model = YOLO(model_dir)  # Thank God Ultralytics manages OpenVINO internally
    def predict(self, frame):
        return self.model(frame, verbose=False)

class NCNNBackend(InferenceBackend):
    def __init__(self, model_path):
        if detect_vulkan():
            device = "NCNN-Vulkan"
        else:
            device = "NCNN-CPU"
        super().__init__(device)
        self.model = YOLO(model_path, task="detect")
    def predict(self, frame):
        return self.model(frame, verbose=False)