import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.applications import VGG16


image_size=224
names=['akari','shiho','kyoko','miho','miku']
num_names=len(names)

# load data
X_train,X_test,Y_train,Y_test = np.load('./imagefiles_224.npy',allow_pickle=True)

Y_train = np_utils.to_categorical(Y_train,num_names)
Y_test = np_utils.to_categorical(Y_test,num_names)
X_train=np.array(X_train)
X_test=np.array(X_test)

# モデルの定義
model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size,image_size,3))

# print('model loaded')
# model.summary()

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(num_names,activation='softmax'))

model = Model(inputs=model.input, outputs=top_model(model.output))
# model.summary()

for layer in model.layers[:15]:
    layer.trainable = False


opt = Adam(0.0001)

model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, epochs=30)

score = model.evaluate(X_test, Y_test, batch_size=32)

model.save('./vgg16_transfer.h5')
