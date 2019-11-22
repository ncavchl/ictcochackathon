import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from sklearn.utils import shuffle

img_size = 201
feature = img_size*img_size
lenclass = 3

TEST_DIR = 'C:/Users/iwsl1/resnet_rooftest/classification_res_test'
test_folder_list = array(os.listdir(TEST_DIR))
print("test_folder_list: ")
print(test_folder_list)

test_input = []
test_label = []

label_encoder = LabelEncoder()  # LabelEncoder Class 호출
# 폴더 리스트 ... 문자형->숫자형
integer_encoded = label_encoder.fit_transform(test_folder_list)
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

for index in range(len(test_folder_list)):
    path = os.path.join(TEST_DIR, test_folder_list[index])
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
        test_input.append([np.array(img)])
        test_label.append([np.array(onehot_encoded[index])])

test_input = np.reshape(test_input, (-1, feature))
test_label = np.reshape(test_label, (-1, lenclass))
test_input = np.array(test_input).astype(np.float32)
test_label = np.array(test_label).astype(np.float32)

train_input,train_label = shuffle(test_input,test_label)
np.save("test_data.npy", test_input)
np.save("test_label.npy", test_label)