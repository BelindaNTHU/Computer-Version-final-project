# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 07:59:29 2023

@author: a3311
"""

import librosa
audio_data = 'audio/ambulance1.mp3'
x , sr = librosa.load(audio_data)
print(x.shape, sr)#(220207,) 22050
librosa.load(audio_data, sr=44100)
import IPython.display as ipd
ipd.Audio(audio_data)

###############################################

import matplotlib.pyplot as plt
import matplotlib
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveshow(x, sr=sr)
plt.savefig("time_domain/timeD_ambulance1.png") #儲存圖片


X = librosa.stft(x) #stft轉換的function，可以用參數調整
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log') #photo2 經stft轉換後的圖
plt.savefig("frequency_domain/frequencyD_ambulance1.png") #儲存圖片


