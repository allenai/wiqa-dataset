import json
import time
import datetime
import numpy as np

# Function to calculate the accuracy of our predictions vs labels
from typing import List, Dict


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def read_jsonl(input_file: str) -> List[Dict]:
    output: List[Dict] = []
    with open(input_file, 'r') as open_file:
        for line in open_file:
            output.append(json.loads(line))
    return output