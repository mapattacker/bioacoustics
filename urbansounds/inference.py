import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from preprocess import preprocess


P = preprocess()
model = load_model("model/")
with open("mapping.json") as f:
    mapping = json.load(f)


def predict(file):
    # extract features
    mfccs_scaled_features = P.mfcc_extractor(file)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)

    # get probability of each class, & convert to class name
    predicted = model.predict(mfccs_scaled_features)
    predicted_label = np.argmax(predicted, axis=1)[0]
    
    # map to class name
    prediction_class = mapping[str(predicted_label)]

    return prediction_class


if __name__ == "__main__":
    file = "/Users/siyang/Downloads/fold1/7383-3-0-0.wav"
    x = predict(file)
    print(x)
