import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras import optimizers
from os import listdir

number = 100
path_bouzu = [filename for filename in listdir('./bouzu_face/') if not filename.startswith('.')]
path_long = [filename for filename in listdir('./long_face/') if not filename.startswith('.')]
path_mash = [filename for filename in listdir('./mash_face/') if not filename.startswith('.')]
path_pama = [filename for filename in listdir('./pama_face/') if not filename.startswith('.')]
path_two_block = [filename for filename in listdir('./two_block_face/') if not filename.startswith('.')]

img_bouzu = []
img_long = []
img_mash = []
img_pama = []
img_two_block = []

hair_list=['bouzu_face', 'long_face', 'mash_face', 'pama_face', 'two_block_face']
path_list=[path_bouzu, path_long, path_mash, path_pama, path_two_block]
list_list=[img_bouzu, img_long, img_mash, img_pama, img_two_block]

for i in range(len(path_list)):
   for j in range(len(path_list[i])):
      try :
         img = cv2.imread('./'+hair_list[i]+'/'+ path_list[i][j])
         img = cv2.resize(img,(64,64))
         list_list[i].append(img)
      except :
         print("{}の読み込み失敗".format(path_list[i][j]) )


np.random.seed(60)
X = np.array(img_bouzu + img_long + img_mash + img_pama + img_two_block)
y =  np.array([0]*len(img_bouzu) + [1]*len(img_long) + [2]*len(img_mash)+ [3]*len(img_pama)+ [4]*len(img_two_block))
rand_index = np.random.permutation(np.arange(len(X)))
X = X[rand_index]
y = y[rand_index]
X_train = X[:int(len(X)*0.8)]
y_train = y[:int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input_tensor = Input(shape=(64, 64, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation="sigmoid"))
top_model.add(Dropout(0.5))
top_model.add(Dense(64, activation='sigmoid'))
top_model.add(Dropout(0.5))
top_model.add(Dense(32, activation='sigmoid'))
top_model.add(Dropout(0.5))
top_model.add(Dense(5, activation='softmax'))
model = Model(input=vgg16.input, output=top_model(vgg16.output))
for layer in model.layers[:15]:
   layer.trainable = False
model.compile(loss='categorical_crossentropy',
             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
             metrics=['accuracy'])

#グラフ用
history = model.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(X_test, y_test))
score = model.evaluate(X_test, y_test, batch_size=128, verbose=0)
print('validation loss:{0[0]}\nvalidation accuracy:{0[1]}'.format(score))
#acc, val_accのプロット
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()
#モデルを保存
model.save("model.h5")





