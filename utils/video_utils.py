import cv2
import os

def extract_and_show_frames(video_path, save_dir="outputs/frames", max_frames=5):
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_path = os.path.join(save_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_path, frame)

        cv2.imshow("Poultry Video Frame", frame)
        cv2.waitKey(500)

    cap.release()
    cv2.destroyAllWindows()

    print(f"{frame_count} frames extracted and displayed")


if __name__ == "__main__":
    extract_and_show_frames("data/poultry.mp4")
