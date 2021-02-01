import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from PIL import Image
import sys

# パラメーターの初期化
names=['akari','shiho','kyoko','miho','miku']
num_classes = len(names)
image_size = 224

# 引数から画像ファイルを参照して読み込む
image = Image.open(sys.argv[1])
image = image.convert('RGB')
image = image.resize((image_size,image_size))
data = np.asarray(image)
X=[]
X.append(data)
X = np.array(X)


# モデルのロード
model = load_model('./vgg16_transfer.h5')
result = model.predict([X])[0]
predicted = result.argmax()
percentage = int(result[predicted]*100)

print(names[predicted], percentage)