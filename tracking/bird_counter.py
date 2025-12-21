import cv2
import numpy as np
import json
from ultralytics import YOLO
from tracking.sort import Sort

def count_birds(video_path, output_json):
    model = YOLO("yolov8n.pt")
    tracker = Sort()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    unique_bird_ids = set()
    frame_number = 0
    timeline_counts = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        time_sec = frame_number / fps

        detections = []
        results = model(frame, conf=0.4)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2])

        if len(detections) == 0:
            tracked_objects = []
        else:
            detections = np.array(detections)
            tracked_objects = tracker.update(detections)

        for obj in tracked_objects:
            bird_id = int(obj[4])
            unique_bird_ids.add(bird_id)

        timeline_counts.append({
            "time_sec": round(time_sec, 2),
            "current_visible_birds": len(tracked_objects),
            "total_unique_birds": len(unique_bird_ids)
        })

    cap.release()

    with open(output_json, "w") as f:
        json.dump(timeline_counts, f, indent=4)

    print("Bird counting completed")
    print("Total unique birds:", len(unique_bird_ids))


if __name__ == "__main__":
    count_birds(
        video_path="data/poultry.mp4",
        output_json="outputs/bird_count.json"
    )
