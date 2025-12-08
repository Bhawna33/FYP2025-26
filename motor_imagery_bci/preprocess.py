import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, low=8, high=30, fs=250):
    
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    filtered = filtfilt(b, a, data)
    return filtered

def preprocess_dataset(X):
  

    X_f = np.zeros_like(X)

    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            X_f[i, ch] = bandpass_filter(X[i, ch])

    mean = X_f.mean(axis=2, keepdims=True)
    std = X_f.std(axis=2, keepdims=True) + 1e-6
    X_norm = (X_f - mean) / std

    return X_norm
