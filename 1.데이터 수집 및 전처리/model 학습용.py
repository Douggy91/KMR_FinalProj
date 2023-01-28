import cv2
import mediapipe as mp
import numpy as np
import time, os, re, csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


file_list = os.listdir("./dataset/")
np_process_list=[]
for i in file_list:
    if i.startswith('seq') == True:
        np_process_list.append(i)
    else: 
        continue
np_need=[]
for i in np_process_list:
    np_need.append(np.load("./dataset/"+i, allow_pickle=True))
    #print(np.load("./dataset/"+i, allow_pickle=True).shape)
data = np.concatenate(np_need, axis=0)
data.shape

file_list_new= []
for i in np_process_list:
    file_list_new.append(i.split('_')[1])
file_list_new=list(set(file_list_new))

x_data = data[:,:,:-1]
labels = data[:,0,-1]

y_data = to_categorical(labels, num_classes=len(file_list_new))

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)


model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(file_list_new), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])


history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=200,
    callbacks=[
        ModelCheckpoint('./models/model.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto'),
        ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=50, verbose=1, mode='auto')
    ]
)

with open('actions_list.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(file_list_new)