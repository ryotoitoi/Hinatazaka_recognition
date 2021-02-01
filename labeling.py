import os
import cv2
import numpy as np
names=['akari','shiho','kyoko','miho','miku']

# 教師データのラベル付け
X_train=[]
Y_train=[]

for i in range(len(names)):
    img_file_name_list=os.listdir('./train/'+names[i])
    print(len(img_file_name_list))
    for j in range(len(img_file_name_list)-1):
        n=os.path.join('./train/'+names[i]+'/',img_file_name_list[j])
        img=cv2.imread(n)
        b,g,r=cv2.split(img)
        img=cv2.merge([r,g,b])
        X_train.append(img)
        Y_train.append(i)

# テストデータのラベル付け
X_test=[]
Y_test=[]
for i in range(len(names)):
    img_file_name_list=os.listdir('./test/'+names[i])
    print(len(img_file_name_list))
    for j in range(0,len(img_file_name_list)-1):
        n=os.path.join('./test/'+names[i]+'/',img_file_name_list[j])
        img = cv2.imread(n)
        b,g,r=cv2.split(img)
        img=cv2.merge([r,b,g])
        X_test.append(img)
        Y_test.append(i)

xy = (X_train,X_test,Y_train,Y_test )
np.save('./imagefiles_224.npy', xy) 
