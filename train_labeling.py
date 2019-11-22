import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from sklearn.utils import shuffle
import tensorflow as tf

img_size = 201
feature = img_size*img_size
lenclass = 3

TRAIN_DIR = "C:/Users/iwsl1/resnet_rooftest/classification_res/"
train_folder_list = array(os.listdir(TRAIN_DIR))
print("train_folder_list: ")
print(train_folder_list)

train_input = []
train_label = []

label_encoder = LabelEncoder()  # LabelEncoder Class 호출
# 폴더 리스트 ... 문자형->숫자형
integer_encoded = label_encoder.fit_transform(train_folder_list)
print("integer_encoded: ")
print(integer_encoded)

onehot_encoder = OneHotEncoder(sparse=False)
# (n,) -> (n,1)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

print("integer_encoded: ")
print(integer_encoded)
print("onehot_encoded: ")
print(onehot_encoded)

for index in range(len(train_folder_list)):
    path = os.path.join(TRAIN_DIR, train_folder_list[index])
    path = path + '/'
    img_list = os.listdir(path)
    print("img_list: ")
    print(len(img_list))
    for img in img_list:
        img_path = os.path.join(path, img)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if index==0:
            img = cv2.resize(img,dsize=(img_size,img_size), interpolation=cv2.INTER_AREA)
        #dst = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_AREA)
        train_input.append([np.array(img)])
        train_label.append([np.array(onehot_encoded[index])])


train_input = np.reshape(train_input, (-1, feature))
train_label = np.reshape(train_label, (-1, lenclass))
train_input = np.array(train_input).astype(np.float32)
train_label = np.array(train_label).astype(np.float32)


train_input,train_label = shuffle(train_input,train_label)
np.save("train_data.npy", train_input)
np.save("train_label.npy", train_label)
