import numpy as np
from detoxify import Detoxify


def detect_toxic_data(doc):
    toxic_data_float64_dict = Detoxify('original').predict(doc.text)

    toxic_data = {}

    for classifier, result in toxic_data_float64_dict.items():
        toxic_data[classifier] = np.float32(result)

    return toxic_data_float64_dict


def detect_toxicity_ratio(scores):
    return (scores["toxic"] + scores["severe_toxic"] + scores["obscene"] +
            scores["threat"] + scores["insult"] + scores["identity_hate"]) / 6
