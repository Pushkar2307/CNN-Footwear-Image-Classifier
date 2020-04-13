from __future__ import print_function
from sklearn.model_selection import train_test_split
from skimage import data, io, filters

from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Conv2DTranspose, UpSampling2D, Reshape, Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K

from keras import utils as np_utils
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import glob
import cv2
import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np
import keras


import keras.utils


#cv2.imwrite("t"+str(i)+"_4.jpg",im)
#i=i+1
im=[]
labels=[]
no_img_train=1808
for name in glob.glob('/home/pushkar/Desktop/Data_test1/*.jpg'):
img=cv2.imread(name,0)
img=cv2.resize(img,(32,32))
#print(img)
names=name.split("/")
names=names[-1]
names=names.split(".")
names=names[0]
names=names.split("_")
names=names[-1]
labels.append(names)
edges=filters.sobel(img)
edges[edges>0.2]=1
edges[edges<=0.2]=255
edges[edges==1]=0
im.append(edges)

#print(labels)
#print(im)
onehot=[0,0,0,0]
onehot_mat=[]
#print(len(labels))
for i in range(len(labels)):
if(labels[i]=='1'):
onehot=[1,0,0,0]
onehot_mat.append(onehot)
if(labels[i]=='2'):
onehot=[0,1,0,0]
onehot_mat.append(onehot)
if(labels[i]=='3'):
onehot=[0,0,1,0]
onehot_mat.append(onehot)
if(labels[i]=='4'):
onehot=[0,0,0,1]
onehot_mat.append(onehot)

model=load_model('/home/pushkar/Desktop/NUSModel/model.h5')
# model = Sequential()
# model.add(Conv2D(3, kernel_size=(3, 3),activation='relu',input_shape=(32,32,1)))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(Conv2D(32, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4, activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

y_train_np=np.array(im)
y_train_np=y_train_np.reshape(len(im),32,32,1)
x_train_fin=np.array(onehot_mat)
#x_train_fin=np.array(onehot_mat).reshape(1,32,32,1)

X_train, X_test, Y_train, Y_test = train_test_split(y_train_np, x_train_fin, test_size=0.1)
#print(X_test)
#model.fit(X_train,Y_train,epochs=10,verbose=1,validation_data=(X_test,Y_test))
#model.save('/home/sartaj/Desktop/NUSModel/model.h5')
pred=model.predict(X_test,verbose=0)

for i in range(len(pred)):
ind=np.argmax(pred[i])
pred[i]=[0,0,0,0]
pred[i][ind]=1

correct=0
incorrect=0

for i in range(len(pred)):
c=0
for j in range(4):
if(pred[i][j]==Y_test[i][j]):
c=c+1
if(c==4):
correct=correct+1
else:
incorrect=incorrect+1

print(Y_test)
print(pred)
print(correct/(correct+incorrect))
#print(labels)
#print(onehot_mat)
