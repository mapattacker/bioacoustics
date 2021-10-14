"""refactored from https://www.kaggle.com/prajaktaparate14/audio-classification"""

import os

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm



class preprocess:

    def mfcc_extractor(self, file):
        """convert audio file into mfcc features"""
        audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
        return mfccs_scaled_features


    def pipeline(self, audio_dataset_path, metadata_file):
        """preprocess pipeline
        urbansounds metadata consists of the following columns
        ["slice_file_name","fsID","start","end","salience","fold","classID","class"]"""

        # load metadata
        self.audio_dataset_path = audio_dataset_path
        metadata_path = os.path.join(audio_dataset_path, metadata_file)
        self.metadata = pd.read_csv(metadata_path)

        extracted_features=[]
        for index_num, row in tqdm(metadata.iterrows()):
            file_name = os.path.join(audio_dataset_path, f'fold{row["fold"]}', row["slice_file_name"])
            final_class_labels = row["class"]
            data = self.mfcc_extractor(file_name)
            extracted_features.append([data,final_class_labels])

        ## Converting extracted_features to pandas dataframe
        extracted_features_df = pd.DataFrame(extracted_features, columns=["feature","class"])

        X = np.array(extracted_features_df["feature"].tolist())
        y = np.array(extracted_features_df["class"].tolist())

        ## Train Test Split
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    audio_dataset_path = "/kaggle/input/urbansound8k"
    metadata_file = "UrbanSound8K.csv"
    P = preprocess()
    X_train, X_test, y_train, y_test = P.pipeline(audio_dataset_path, metadata_file)