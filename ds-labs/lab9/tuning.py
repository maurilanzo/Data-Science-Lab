import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import os
import scipy.io.wavfile as wavfile
from scipy.signal import resample

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

# Funzione di tuning: padding prima della FFT + addestramento RandomForest
def tuner(target_length, **rf_params):
    wavs_dev = os.listdir("free-spoken-digit/dev")
    wavlist_dev = sorted(wavs_dev, key=lambda x: int(x.split("_")[0]))
    
    X_dev = []
    y_dev = []

    TARGET_LENGTH = target_length  # Numero di campioni da tagliare/paddare PRIMA della FFT

    for file in wavlist_dev:
        path = os.path.join("free-spoken-digit/dev", file)

        parts = file.split('_')
        label = int(parts[1].split('.')[0])

        y_dev.append(label)
        
        rate, signal = wavfile.read(path)

        if len(signal) > TARGET_LENGTH:
            signal = signal[:TARGET_LENGTH]
        else:
            pad_width = TARGET_LENGTH - len(signal)
            signal = np.pad(signal, (0, pad_width), mode='constant')

        # FFT
        fft_val = np.abs(np.fft.fft(signal))
        X_dev.append(fft_val)

    X = pd.DataFrame(X_dev)
    y = np.array(y_dev)  # vettore 1D va benissimo per sklearn

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    # Assicuriamoci che alcuni parametri di default ci siano sempre
    rf_params = dict(rf_params)
    rf_params.setdefault("random_state", 42)
    rf_params.setdefault("n_jobs", -1)

    forest = RandomForestClassifier(**rf_params)
    forest.fit(X_train, y_train)
    y_pred = forest.predict(X_val)

    return accuracy_score(y_val, y_pred)