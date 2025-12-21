import cv2
from ultralytics import YOLO

def detect_birds(video_path, output_path):
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.4)

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if model.names[cls_id] == "bird":
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Bird {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

        out.write(frame)

    cap.release()
    out.release()

    print("Bird detection video saved successfully")


if __name__ == "__main__":
    detect_birds(
        video_path="data/poultry.mp4",
        output_path="outputs/bird_detection.mp4"
    )
