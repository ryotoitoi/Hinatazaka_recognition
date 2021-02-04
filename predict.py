import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image
import sys
import cv2,os

# パラメーターの初期化
names=['akari','shiho','kyoko','miho','miku']
num_classes = len(names)
image_size = 224

# 引数から画像ファイルを参照して読み込む
# image = Image.open(sys.argv[1])
image=cv2.imread(sys.argv[1])
image_gs=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_alt.xml')
# 顔認識の実行
face_list=cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2,minSize=(64,64))       
#顔が１つ以上検出された時
if len(face_list) > 0:
    for rect in face_list:
        x,y,width,height=rect
        image = image[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
        image=cv2.resize(image,(image_size,image_size))
        print('We can detect your face!')
#顔が検出されなかった時
else:
    print("no face")

data = np.asarray(image)
X=[]
X.append(data)
X = np.array(X)


# モデルのロード
model = load_model('./vgg16_transfer.h5')
result = model.predict([X])[0]
predicted = result.argmax()
percentage = int(result[predicted]*100)

print(result)
print(names[predicted], percentage)