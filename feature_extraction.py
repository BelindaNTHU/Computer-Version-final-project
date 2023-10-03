# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 07:59:29 2023

@author: a3311
"""

import librosa
audio_data = 'road1.mp3'
x , sr = librosa.load(audio_data)
print(type(x), type(sr))# print(x.shape, sr)#(94316,) 22050
librosa.load(audio_data, sr=44100)
import IPython.display as ipd
ipd.Audio(audio_data)

import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)

X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') #y_axis='log'
plt.colorbar()
     
#Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)

# Zooming in
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
