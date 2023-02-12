import numpy as np
from detoxify import Detoxify


def detect_toxic_data(nlp_doc):
    toxic_data_float64_dict = Detoxify('original').predict(nlp_doc.text)

    toxic_data = {}

    for classifier, result in toxic_data_float64_dict.items():
        toxic_data[classifier] = np.float(result)

    return toxic_data
