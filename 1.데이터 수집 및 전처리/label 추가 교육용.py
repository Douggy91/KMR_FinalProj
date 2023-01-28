import cv2
import mediapipe as mp
import numpy as np
import time, os, re, csv
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# 기존 입력값이 있는 list 만들기
# actions_pre= 기존 입력 값들

# list 만들기

with open('actions_list.csv', 'r', encoding='utf-8') as f:
    rdr = csv.reader(f)
    for i,line in enumerate(rdr):
        if i==0:
            actions_pre = line
            
inac=input('동작이름을 입력하세요: ')

if inac not in actions_pre:
    actions_pre.append(inac)
else:
    print('동작이 중복되었습니다! 다시시도해주세요!')
# csv파일로 만들기

    
file_list = os.listdir('./dataset')

# file_list로 등록된 라벨들 추출

file_list_new= []
for i in file_list:
    file_list_new.append(i.split('_')[1])
    
# actions 중에서 현재 파일 목록에 없는 놈만 추출    

actions_new= []
for i in actions_pre:
    for v in file_list_new:
        if i in v:
            actions_new.append(i)
            
actions=list(set(actions_pre)-set(actions_new))
seq_length = 60
secs_for_action = 60

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    angle = np.degrees(angle) # Convert radian to degree

                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break

        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break

# 새로 학습한 라벨 new_actions_list.csv로 저장 

with open('new_actions_list.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(actions)

file_list = os.listdir('./dataset')
np_process_list=[]
for i in file_list:
    if i.startswith('seq') == True:
        np_process_list.append(i)
    else: 
        continue

# 추가한 학습 데이터만 불러오기 
    
# np_process_comp=[]    

# for i in np_process_list:
#     if inac in i:
#         np_process_comp.append(i)

np_need=[]
for i in np_process_list:
    np_need.append(np.load("./dataset/"+i, allow_pickle=True))

data = np.concatenate(np_need, axis=0)
data.shape


x_data = data[:,:,:-1]
labels = data[:,0,-1]

y_data = to_categorical(labels, num_classes=len(actions_pre))

x_data = x_data.astype(np.float32)
y_data = y_data.astype(np.float32)

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=2021)


model = Sequential([
    LSTM(64, activation='relu', input_shape=x_train.shape[1:3]),
    Dense(32, activation='relu'),
    Dense(len(actions_pre), activation='softmax')
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
    writer.writerow(actions_pre)