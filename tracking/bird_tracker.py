import cv2
import numpy as np
from ultralytics import YOLO
from tracking.sort import Sort

def track_birds(video_path, output_path):
    model = YOLO("yolov8n.pt")
    tracker = Sort()

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = []
        results = model(frame, conf=0.4)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if model.names[cls_id] == "bird":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2])

        detections = np.array(detections)
        tracked = tracker.update(detections)

        for obj in tracked:
            x1, y1, x2, y2, track_id = map(int, obj)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"ID {track_id}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2
            )

        out.write(frame)

    cap.release()
    out.release()
    print("Tracked bird video saved successfully")


if __name__ == "__main__":
    track_birds(
        video_path="data/poultry.mp4",
        output_path="outputs/bird_tracking.mp4"
    )
