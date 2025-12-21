# Bird Counting and Weight Estimation using Computer Vision

## Overview
This project is a prototype system that processes a fixed-camera poultry farm CCTV video to:
- Count birds over time using object detection and stable tracking IDs
- Estimate bird weight using a proxy-based method
- Expose results through a FastAPI service

The project demonstrates practical knowledge of computer vision, tracking, analytics, and API development.

---

## Project Pipeline
1. Input fixed-camera poultry CCTV video
2. Bird detection using YOLOv8
3. Bird tracking using SORT (stable IDs)
4. Bird counting over time using tracking IDs
5. Bird weight estimation using bounding box area as a proxy
6. Analytics aggregation
7. REST API using FastAPI



## Folder Structure
bird-counting-project/
│
├── data/
│ └── poultry.mp4
│
├── detection/
│ └── bird_detector.py
│
├── tracking/
│ ├── sort.py
│ ├── bird_tracker.py
│ └── bird_counter.py
│
├── weight_estimation/
│ └── weight_estimator.py
│
├── utils/
│ ├── video_utils.py
│ └── analytics.py
│
├── api/
│ └── app.py
│
├── outputs/
│ ├── bird_count.json
│ ├── bird_weights.json
│ ├── analytics_summary.json
│ └── final_annotated_video.mp4
│
├── requirements.txt
└── README.md

## Technologies Used
- Python
- OpenCV
- YOLOv8 (Ultralytics)
- SORT Tracking (Kalman Filter + IOU)
- NumPy
- FastAPI
- Uvicorn

---

## Bird Detection
Birds are detected in each video frame using a pretrained YOLOv8 model.  
Bounding boxes are extracted for detected objects and passed to the tracker.

---

## Bird Tracking
SORT (Simple Online and Realtime Tracking) is used to assign stable IDs to birds across frames.  
This ensures the same bird keeps the same ID and avoids double counting.

---

## Bird Counting Logic
- Each new tracking ID is counted as a new bird
- Repeated IDs are ignored
- Bird counts are recorded over time and stored in JSON format

---

## Bird Weight Estimation (Proxy-Based)
Since real bird weight cannot be measured from video:
- Bounding box area is used as a proxy for bird size
- Average area over time is calculated per bird
- A scaling factor is applied to estimate weight in grams

This provides a relative weight index rather than an exact measurement.

---

## API Service (FastAPI)

### Start the API Server

uvicorn api.app:app --reload
API Endpoints
GET / → API health check

GET /bird-count → Bird count over time

GET /bird-weights → Weight estimate per bird

GET /analytics → Aggregated summary statistics

Swagger UI
arduino
Copy code
http://127.0.0.1:8000/docs
Outputs
Annotated output video with bird ID and weight

JSON files for bird counts, weights, and analytics

REST API responses via FastAPI

Notes
Weight estimation is proxy-based and not a real physical measurement

The system assumes a fixed camera

The focus is on correct methodology rather than perfect accuracy