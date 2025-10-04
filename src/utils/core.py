# src/utils/core.py
import cv2
import time
import os
from src.utils.visualization import draw_boxes
from src.utils.helpers import save_metrics
from src.utils.backends import InferenceBackend, TorchBackend, OpenVINOBackend, NCNNBackend

OUTPUT_DIR = "results/sample_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_camera(backend: InferenceBackend, max_duration: int = 10):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Couldn't open camera")
        return

    print(f"Starting camera inference for {max_duration} seconds... (Press Ctrl+C to stop early)")

    # Prepare video output
    OUTPUT_DIR = "results/sample_outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "output_camera.mp4")

    fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()
    inference_times = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Stop automatically after max_duration seconds
            if time.time() - start_time > max_duration:
                print(f"Max duration ({max_duration}s) reached.")
                break

            # Perform inference
            t0 = time.perf_counter()
            results = backend.predict(frame)
            t1 = time.perf_counter()
            inference_time = (t1 - t0) * 1000
            inference_times.append(inference_time)

            # Draw detections
            frame = draw_boxes(frame, results)
            out.write(frame)
            frame_count += 1

    finally:
        cap.release()
        out.release()

    # Calculate metrics
    avg_fps = frame_count / (time.time() - start_time)
    avg_inf_time = sum(inference_times) / len(inference_times)

    print(f"Camera closed. Avg FPS: {avg_fps:.2f}, Avg Inference: {avg_inf_time:.1f} ms")
    print(f"Saved annotated video to: {out_path}")

    save_metrics({
        "mode": "camera",
        "device": backend.device,
        "frames": frame_count,
        "duration_s": max_duration,
        "avg_fps": avg_fps,
        "avg_inference_time_ms": avg_inf_time,
        "output_video": out_path
    })

def select_file_headless():
    """Fallback for Docker (no GUI available)."""
    print("Running inside Docker: GUI file selection disabled.")
    print("Please provide the file path manually or use the API instead.")
    return None

def run_file(backend: InferenceBackend):
    # Detect if running inside Docker
    in_docker = os.path.exists("/.dockerenv")

    if in_docker:
        file_path = select_file_headless()
        if not file_path:
            print("❌ No file selected (Docker environment)")
            return
    else:
        from tkinter import Tk, filedialog
        Tk().withdraw()
        file_path = filedialog.askopenfilename(
            title="Choose a video or image",
            filetypes=[("Supported files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if not file_path:
            print("❌ No file received")
            return

    cap = cv2.VideoCapture(file_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(OUTPUT_DIR, "output_video.mp4")
    fps_in = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps_in, (width, height))

    frame_count = 0
    start_time = time.time()
    inference_times = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
        results = backend.predict(frame)
        t1 = time.perf_counter()
        inference_time = (t1 - t0) * 1000
        inference_times.append(inference_time)

        frame = draw_boxes(frame, results)
        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    avg_fps = frame_count / (time.time() - start_time)
    avg_inf_time = sum(inference_times) / len(inference_times)
    print(f"✅ Done. Average FPS: {avg_fps:.2f}, Average Inference: {avg_inf_time:.1f} ms. Saved in: {out_path}")

    save_metrics({
        "mode": "video",
        "device": backend.device,
        "file": file_path,
        "frames": frame_count,
        "resolution": f"{width}x{height}",
        "avg_fps": avg_fps,
        "avg_inference_time_ms": avg_inf_time
    })