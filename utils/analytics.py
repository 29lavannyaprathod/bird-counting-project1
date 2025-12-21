import json
import statistics

def generate_analytics(
    count_json="outputs/bird_count.json",
    weight_json="outputs/bird_weights.json",
    output_json="outputs/analytics_summary.json"
):
    with open(count_json, "r") as f:
        count_data = json.load(f)

    with open(weight_json, "r") as f:
        weight_data = json.load(f)

    total_unique_birds = count_data[-1]["total_unique_birds"]
    max_visible_birds = max(item["current_visible_birds"] for item in count_data)

    weights = list(weight_data.values())

    analytics = {
        "total_unique_birds": total_unique_birds,
        "max_visible_birds": max_visible_birds,
        "average_weight_grams": round(statistics.mean(weights), 2),
        "min_weight_grams": round(min(weights), 2),
        "max_weight_grams": round(max(weights), 2),
        "per_bird_weights": weight_data
    }

    with open(output_json, "w") as f:
        json.dump(analytics, f, indent=4)

    print("Analytics summary generated successfully")


if __name__ == "__main__":
    generate_analytics()
