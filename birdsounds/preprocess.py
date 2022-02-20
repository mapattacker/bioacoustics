"""refactored from the kaggle notebook
https://www.kaggle.com/shtrausslearning/keras-bird-spectogram-multiclass-classification/notebook
"""

import pandas as pd
import librosa
from tqdm import tqdm, tnrange, tqdm_notebook



def split_signal(sig, sr, sl):
    """Split signal into equal segments
    
    sig (array): 
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


def save_melspec():
    """save mel spectrogram"""


def get_spectrograms(filepath, primary_label, output_dir):
    """extracts spectrograms and saves them in a working directory"""

    # duration is set from global variable
    sig, rate = librosa.load(filepath, sr=coefs.sr, offset=None, duration=coefs.cutoff)
    sig_splits = split_signal(sig) # split the signal into parts
    
    # Extract mel spectrograms for each audio chunk
    s_cnt = 0
    saved_samples = []
    for chunk in sig_splits:
        
        hop_length = int(coefs.sl * coefs.sr / (coefs.sshape[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=chunk, 
                                                  sr=coefs.sr, 
                                                  n_fft=1024, 
                                                  hop_length=hop_length, 
                                                  n_mels=coefs.sshape[0], 
                                                  fmin=coefs.fmin, 
                                                  fmax=coefs.fmax)
    
        mel_spec = librosa.power_to_db(mel_spec**2, ref=np.max) 
        
        # Normalize
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()
        
        # Save as image file
        save_dir = os.path.join(output_dir, primary_label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filepath.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] + 
                                 '_' + str(s_cnt) + '.png')
        im = Image.fromarray(mel_spec * 255.0).convert("L")
        im.save(save_path)
        
        saved_samples.append(save_path)
        s_cnt += 1
        
    return saved_samples



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
