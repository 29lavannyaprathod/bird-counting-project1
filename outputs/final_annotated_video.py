import cv2
import json
import numpy as np
from ultralytics import YOLO
from tracking.sort import Sort

def generate_final_video(video_path, weights_json, output_video):
    model = YOLO("yolov8n.pt")
    tracker = Sort()

    with open(weights_json, "r") as f:
        bird_weights = json.load(f)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter(
        output_video,
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
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2])

        if len(detections) > 0:
            detections = np.array(detections)
            tracked = tracker.update(detections)
        else:
            tracked = []

        for obj in tracked:
            x1, y1, x2, y2, bird_id = map(int, obj)
            weight = bird_weights.get(str(bird_id), "NA")

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {bird_id} | {weight} g",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        cv2.putText(
            frame,
            f"Total Birds: {len(tracked)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )

        out.write(frame)

    cap.release()
    out.release()
    print("Final annotated video generated successfully")


if __name__ == "__main__":
    generate_final_video(
        video_path="data/poultry.mp4",
        weights_json="outputs/bird_weights.json",
        output_video="outputs/final_annotated_video.mp4"
    )
