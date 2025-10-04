import cv2 

def draw_boxes(frame, results):
    for box in results[0].boxes:
           x1, y1, x2, y2 = map(int, box.xyxy[0])
           conf = float(box.conf)
           cls = int(box.cls)
           label = f"{results[0].names[cls]} {conf:.2f}"
           cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
           cv2.putText(frame, label, (x1, y1 - 5),
           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame