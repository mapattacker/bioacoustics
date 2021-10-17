import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

from preprocess import preprocess


P = preprocess()
model = load_model("model/")
labelencoder = joblib.load("encoder/encoder.jb")


def predict(file):
    # extract features
    mfccs_scaled_features = P.mfcc_extractor(file)
    mfccs_scaled_features = mfccs_scaled_features.reshape(1,-1)

    # get probability of each class, & convert to class name
    predicted = model.predict(mfccs_scaled_features)
    predicted_label = np.argmax(predicted, axis=1)
    prediction_class = labelencoder.inverse_transform(predicted_label)

    return prediction_class


if __name__ == "__main__":
    file = "/Users/siyang/Downloads/fold1/7383-3-0-0.wav"
    x = predict(file)
    print(x)