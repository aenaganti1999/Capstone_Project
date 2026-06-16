import json
from pathlib import Path

PREDICTION_FILE = "predictions.json"


def save_prediction(prediction_data):

    file_path = Path(PREDICTION_FILE)

    if file_path.exists():

        with open(PREDICTION_FILE, "r") as f:
            predictions = json.load(f)

    else:
        predictions = []

    predictions.append(prediction_data)

    with open(PREDICTION_FILE, "w") as f:
        json.dump(
            predictions,
            f,
            indent=4
        )