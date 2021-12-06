"""refactored from the kaggle notebook
https://www.kaggle.com/shtrausslearning/keras-bird-spectogram-multiclass-classification/notebook
"""

import pandas as pd
import librosa
from tqdm import tqdm, tnrange, tqdm_notebook



def split_signal(sig, sr, sl):
    """Split signal into equal segments
    
    sr (int): sampling rate
    sl (int) sampling length"""
    step = int(sample_rate * sample_len)
    sig_splits = []
    for i in range(0, len(sig), step):
        split = sig[i:i + step]
        if len(split) < step:
            break
        sig_splits.append(split)
    return sig_splits




if __name__ == "__main__":
    import yaml
    f = open("conf/config.yaml")
    a = yaml.safe_load(f)
    # print(a)


    filepath = "/Users/siyang/Downloads/XC109605.ogg"
    filepath = "/Users/siyang/Downloads/XC110258.ogg"
    
    sig, rate = librosa.load(filepath, offset=None, duration=28)
    print(rate)
    # sample_rate = 32000
    # sample_len = 5
    # a = split_signal(sig, sample_rate, sample_len)
    # print(len(a))