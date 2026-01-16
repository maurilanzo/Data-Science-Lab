import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import os
import scipy.io.wavfile as wavfile
from scipy.signal import resample

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#padding before fft

#PARAMETERS:
max_length = 3000    #Number of samples to trim/pad BEFORE FFT  (also this was tuned)

wavs_dev = ((os.listdir("free-spoken-digit/dev")))
wavlist_dev = sorted(wavs_dev, key=lambda x: int(x.split("_")[0]))
devs = []
lengs = []
X_dev = []
y_dev = []
 

for file in wavlist_dev:
    path = os.path.join("free-spoken-digit/dev", file)      #ex: 0_4.wav

    parts = file.split('_')         #----> 0 , 4.wav
    label = int(parts[1].split('.')[0])     #---> 4 , .wav  and i take 4

    y_dev.append(label)
    
    rate, signal = wavfile.read(path)       #rate = 8000

    if len(signal) > max_length:
        signal = signal[:max_length]
    else:
        pad_width = max_length - len(signal)
        signal = np.pad(signal, (0, pad_width), mode='constant')        #Trim or pad the signal

    
    fft_val = np.abs(np.fft.fft(signal))        #perform fft to have a magnitude spectrum of the signal

    X_dev.append(fft_val)

X_train = pd.DataFrame(X_dev)
y_train = pd.DataFrame(y_dev)


wavs_eval = ((os.listdir("free-spoken-digit/eval")))
wavlist_eval = sorted(wavs_eval, key = lambda x: int(x.split(".")[0]) )
devs=[]
lengs= []
X_eval = []

for file in wavlist_eval:
    path = os.path.join("free-spoken-digit/eval", file)
    parts = int(file.split('.')[0])

    
    (rate, signal) = wavfile.read(path)
    
    if len(signal) > max_length:
        signal = signal[:max_length]
    else:
        pad_width = max_length - len(signal)
        signal = np.pad(signal, (0, pad_width), mode='constant')

    fft_val = np.abs(np.fft.fft(signal))
    X_eval.append(fft_val)     #Does the FFT and resample the signal with 6000 samples

X_test = pd.DataFrame(X_eval)


#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=0.8, random_state=42)

min_samples_leaf= 1
min_samples_split= 2
n_estimators= 200       #chosen by tuning
max_features = "sqrt"
max_depth= 20
forest = RandomForestClassifier(min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split,n_estimators=n_estimators,max_features=max_features,max_depth=max_depth)


forest.fit(X_train,y_train)
y_pred= forest.predict(X_test)
data = {'Id':np.linspace(0,len(y_pred)-1,len(y_pred),dtype="int"),'Predicted':y_pred}
submission = pd.DataFrame(data)
submission.to_csv("submission.csv",index=None)

