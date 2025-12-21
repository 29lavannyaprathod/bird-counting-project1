import cv2
import numpy as np
import json
from ultralytics import YOLO
from tracking.sort import Sort

def estimate_bird_weights(video_path, output_json):
    model = YOLO("yolov8n.pt")
    tracker = Sort()

    cap = cv2.VideoCapture(video_path)

    bird_areas = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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
            x1, y1, x2, y2, bird_id = map(int, obj)
            area = (x2 - x1) * (y2 - y1)

            if bird_id not in bird_areas:
                bird_areas[bird_id] = []

            bird_areas[bird_id].append(area)

    cap.release()

    SCALE_FACTOR = 0.12

    bird_weights = {}
    for bird_id, areas in bird_areas.items():
        avg_area = sum(areas) / len(areas)
        estimated_weight = round(avg_area * SCALE_FACTOR, 2)
        bird_weights[bird_id] = estimated_weight

    with open(output_json, "w") as f:
        json.dump(bird_weights, f, indent=4)

    print("Bird weight estimation completed")
    print("Estimated weights saved")


if __name__ == "__main__":
    estimate_bird_weights(
        video_path="data/poultry.mp4",
        output_json="outputs/bird_weights.json"
    )
