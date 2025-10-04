import os
import cv2
import time
import shutil
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from src.inference import select_backend
from src.utils.helpers import save_metrics
from src.utils.visualization import draw_boxes

# ------------------------------#
#        FastAPI Setup          #
# ------------------------------#
app = FastAPI(
    title="HVAC Detector API",
    description="REST API for automatic HVAC detection using YOLOv11 and multiple inference backends",
    version="1.0.0"
)

# ------------------------------#
#       Initialization          #
# ------------------------------#
backend = select_backend()
print(f"✅ Backend active in API: {backend.device}")

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "results/sample_outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------#
#           Endpoints           #
# ------------------------------#
@app.get("/")
def root():
    """Root endpoint: shows API info"""
    return {
        "message": "HVAC Detector API running successfully 🚀",
        "backend": backend.device,
        "available_endpoints": {
            "POST /predict/file": "Upload an image or video for detection",
            "GET /predict/camera": "Capture from webcam (server-side)",
            "GET /download/{filename}": "Download processed results"
        }
    }


@app.post("/predict/file")
async def predict_file(file: UploadFile = File(...)):
    """Upload image or video for detection"""
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"📂 File received: {file.filename}")
    ext = file.filename.lower().split(".")[-1]

    # --------------------#
    #    IMAGE MODE       #
    # --------------------#
    if ext in ["jpg", "jpeg", "png"]:
        img = cv2.imread(temp_path)
        if img is None:
            return JSONResponse(content={"error": "Invalid image format"}, status_code=400)

        print("🖼️ Image detected — running inference...")
        start = time.time()
        results = backend.predict(img)
        end = time.time()
        inference_time = (end - start) * 1000

        # Draw detections and save
        img_annotated = draw_boxes(img.copy(), results)
        output_filename = f"output_{file.filename}"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, img_annotated)

        # Save metrics
        save_metrics({
            "mode": "image",
            "device": backend.device,
            "filename": file.filename,
            "avg_inference_time_ms": round(inference_time, 2)
        })

        # Prepare detections list
        detections = []
        if hasattr(results[0], "boxes"):
            for box in results[0].boxes:
                detections.append({
                    "class": results[0].names[int(box.cls)],
                    "confidence": round(float(box.conf), 3),
                    "box": list(map(int, box.xyxy[0]))
                })

        return {
            "filename": file.filename,
            "type": "image",
            "device": backend.device,
            "inference_time_ms": round(inference_time, 2),
            "detections": detections,
            "output_file": output_filename
        }

    # --------------------#
    #    VIDEO MODE       #
    # --------------------#
    elif ext in ["mp4", "avi", "mov"]:
        print("🎥 Video detected — running inference...")
        cap = cv2.VideoCapture(temp_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = os.path.join(OUTPUT_DIR, f"output_{file.filename}")
        width, height = 1280, 720
        out = cv2.VideoWriter(out_path, fourcc, 25, (width, height))

        frame_count = 0
        start_time = time.time()
        inference_times = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            t0 = time.perf_counter()
            results = backend.predict(frame)
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) * 1000)
            frame = draw_boxes(frame, results)
            out.write(frame)
            frame_count += 1

        cap.release()
        out.release()

        avg_fps = frame_count / (time.time() - start_time)
        avg_inf_time = sum(inference_times) / len(inference_times)

        save_metrics({
            "mode": "video",
            "device": backend.device,
            "filename": file.filename,
            "frames": frame_count,
            "avg_fps": round(avg_fps, 2),
            "avg_inference_time_ms": round(avg_inf_time, 2)
        })

        return {
            "filename": file.filename,
            "type": "video",
            "frames_processed": frame_count,
            "device": backend.device,
            "avg_fps": round(avg_fps, 2),
            "avg_inference_time_ms": round(avg_inf_time, 2),
            "output_video": os.path.basename(out_path)
        }

    # --------------------#
    #  UNSUPPORTED TYPE   #
    # --------------------#
    else:
        return JSONResponse(
            content={"error": "Unsupported file format"},
            status_code=400
        )


@app.get("/predict/camera")
async def predict_camera(duration: int = 10):
    """Capture live video from the server webcam"""
    print(f"🎦 Starting camera capture for {duration}s...")
    from src.utils.core import run_camera
    run_camera(backend, max_duration=duration)
    return {
        "mode": "camera",
        "device": backend.device,
        "duration_s": duration,
        "output_video": "output_camera.mp4"
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download processed files"""
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    return FileResponse(file_path, media_type="application/octet-stream", filename=filename)