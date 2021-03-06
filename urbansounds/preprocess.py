"""refactored from https://www.kaggle.com/prajaktaparate14/audio-classification"""

import json
import os

import joblib
import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm



class preprocess:

    def mfcc_extractor(self, file):
        """convert audio file into normalised mfcc features"""
        audio, sample_rate = librosa.load(file, res_type="kaiser_fast")
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features


    def f_extractor(self, df, idx):
        """some processing before feature extraction"""
        row = df.iloc[idx]
        file_name = os.path.join(self.audio_dataset_path, f'fold{row["fold"]}', row["slice_file_name"])
        final_class_labels = row["class"]
        data = self.mfcc_extractor(file_name)
        return [data, final_class_labels]


    def encoder(self, y):
        """encode string labels to one-hot & save encode mapping"""
        # encode label string to int
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(y)
        
        # save mapping as json
        mapping = {i: label for i, label in enumerate(labelencoder.classes_)}
        with open("mapping.json", "w") as fp:
            json.dump(mapping, fp)
        
        # one-hot encoding
        y = to_categorical(y)
        return y


    def pipeline(self, audio_dataset_path, metadata_file, n_jobs=4):
        """preprocess pipeline
        urbansounds metadata consists of the following columns
        ["slice_file_name","fsID","start","end","salience","fold","classID","class"]"""

        # load metadata
        self.audio_dataset_path = audio_dataset_path
        metadata_path = os.path.join(self.audio_dataset_path, metadata_file)
        metadata = pd.read_csv(metadata_path)

        # parallel extractor
        extracted_features = Parallel(n_jobs=n_jobs)(
            delayed(self.f_extractor)(metadata, idx) for idx in tqdm(range(len(metadata))))

        ## Converting extracted_features to pandas dataframe
        extracted_features_df = pd.DataFrame(extracted_features, columns=["feature","class"])

        X = np.array(extracted_features_df["feature"].tolist())
        y = np.array(extracted_features_df["class"].tolist())

        y = self.encoder(y)

        ## Train Test Split
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    audio_dataset_path = "/kaggle/input/urbansound8k"
    metadata_file = "UrbanSound8K.csv"
    P = preprocess()
    X_train, X_test, y_train, y_test = P.pipeline(audio_dataset_path, metadata_file)
