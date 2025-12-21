from fastapi import FastAPI
import json

app = FastAPI(title="Bird Counting and Weight Estimation API")

@app.get("/")
def home():
    return {"status": "Bird counting API is running"}

@app.get("/bird-count")
def get_bird_count():
    with open("outputs/bird_count.json", "r") as f:
        data = json.load(f)

    return {
        "total_unique_birds": data[-1]["total_unique_birds"],
        "timeline": data
    }

@app.get("/bird-weights")
def get_bird_weights():
    with open("outputs/bird_weights.json", "r") as f:
        data = json.load(f)

    return {
        "bird_weights_grams": data
    }

@app.get("/analytics")
def get_analytics():
    with open("outputs/analytics_summary.json", "r") as f:
        data = json.load(f)

    return data
