# -*- coding: utf-8 -*-
# @Time    : 2018/12
# @Author  : wengfutian
# @Email   : wengfutian@csu.edu.cn
# Keras for minist classification

# Import librares
import numpy as np
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
#from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling2D, Input, Conv2D
from keras import backend as K
from keras.models import Model, model_from_json
#K.common.set_image_dim_ordering('th')
import json

#Load googlenet architecture
#Google Colab '/content/GoogleNet_Vgg16_thermy/googlenet.json'
#Kaggle
with open('/kaggle/working/GoogleNet_Vgg16_thermy/googlenet.json','r') as f:
    model_json = json.load(f)

#Load googlenet model weights
googlenet_model = model_from_json(json.dumps(model_json))

#Import googlenet model with 'he_uniform' intialization
#Kaggle
googlenet_model.load_weights("/kaggle/working/GoogleNet_Vgg16_thermy/googlenet.h5")
#Google Colab
#googlenet_model.load_weights("/content/GoogleNet_Vgg16_thermy/googlenet.h5")

googlenet_model.summary()

# load model without output layer
#model = googlenet_model(include_top=False)

# load minit data
from keras.datasets import mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

# plot 6 images as gray scale
import matplotlib.pyplot as plt

plt.subplot(321)
plt.imshow(x_train[0],cmap=plt.get_cmap('gray'))
plt.subplot(322)
plt.imshow(x_train[1],cmap=plt.get_cmap('gray'))
plt.subplot(323)
plt.imshow(x_train[2],cmap=plt.get_cmap('gray'))
plt.subplot(324)
plt.imshow(x_train[3],cmap=plt.get_cmap('gray'))
plt.subplot(325)
plt.imshow(x_train[4],cmap=plt.get_cmap('gray'))
plt.subplot(326)
plt.imshow(x_train[5],cmap=plt.get_cmap('gray'))
# show
plt.show()

# reshape the data to four dimensions, due to the input of model
# reshape to be [samples][width][height][pixels]
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')


# normalization
x_train = x_train / 255.0
x_test = x_test / 255.0

# one-hot
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# parameters
EPOCHS = 1
INIT_LR = 1e-4
BS = 32
CLASS_NUM = 10
norm_size = 28

# start to train model
print('start to train model')

#Using Conv2D with he_uniform
#Ricardo Niño de Rivera 10:22 hrs 11/12/2019

# define VGG!& model
def VGG16(width, height, depth, NB_CLASS):
    model = Sequential()
    inputShape = (height, width, depth)
    # Block 1
    model.add(ZeroPadding2D((1,1), input_shape=inputShape))
    model.add(Conv2D(64, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(NB_CLASS))
    model.add(Activation("softmax"))

    return model

#Accedemos a las capas de googlenet_model
print()
print('Prueba1 (imprimiendo salida índice -5)')
print(googlenet_model.layers[-5].output)
print()
print('Prueba2 (imprimiendo salidas índices hasta -5)')
for i in range(0, len(googlenet_model.layers)-4):
    print(i)
    print(googlenet_model.layers[i].input)
    print(googlenet_model.layers[i].output)
    #new_model.add(googlenet_model.layers[i])
print()

#Eliminamos las cuatro primeras últimas de googlenet_model
googlenet_model._layers.pop() #Primera capa eliminada
googlenet_model._layers.pop() #Segunda capa eliminada
googlenet_model._layers.pop() #Tercera capa eliminada
googlenet_model._layers.pop() #Cuarta capa eliminada

#"Congelamos" las capas del modelo
googlenet_model.trainable = False
for layer in googlenet_model.layers:
    layer.trainable = False

#Imprimimos las capas del googlenet_model
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in googlenet_model.layers]
print(pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']))

#Imprimiendo salida de la última capa googlenet_model
print()
print(googlenet_model.layers[-1].output)
size_G_out = googlenet_model.layers[-1].output.shape #Tamaño del vector de salida de la última capa
print(googlenet_model.layers[-1].output.shape)
print(size_G_out[0])
print(size_G_out[1])
print(size_G_out[2])
print(size_G_out[3])
print()

#Creamos model1 sin las últimas 4 capas de googlenet_model
model1 = Model(googlenet_model.input, googlenet_model.layers[-1].output)
model1.summary()

print()
print(model1.output)
print()

#Creamos new_model como modelo secuencial
new_model = Sequential()

#Añadimos model1 a new_model

new_model.add(model1)

## ALTERNATIVA DE IMPLEMENTACIÓN ##

#Añadimos el modelo VGG16 sin utilizar la función definida VGG16 POR PROBLEMAS DE COMPATIBILIDAD
#Block1
#new_model.add(ZeroPadding2D((1,1), input_shape=(size_G_out[1], size_G_out[2], size_G_out[3])))
#new_model.add(Conv2D(64, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(64, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
#Block2
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(128, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(128, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
#Block3
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(256, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(256, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(256, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
#Block4
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
#Block5
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(ZeroPadding2D((1,1)))
#new_model.add(Conv2D(512, kernel_size=3, strides=3, activation='relu', kernel_initializer='he_uniform'))
#new_model.add(MaxPooling2D(pool_size=(2, 2), strides=2, padding='same'))
#Block6
#new_model.add(Flatten())
#new_model.add(Dense(4096, activation='relu'))
#new_model.add(Dense(4096, activation='relu'))
#new_model.add(Dense(10))
#new_model.add(Activation("softmax"))

## TERMINA ALTERNATIVA DE IMPLEMENTACIÓN ##

## IMPLEMENTACIÒN CON MENOS CÒDIGO ##
#Creamos model2
model2 = VGG16(width=size_G_out[1], height=size_G_out[2], depth=size_G_out[3], NB_CLASS=10)
#añadimos model2 a new_model
new_model.add(model2)
## TERMINA IMPLEMENTACIÒN CON MENOS CÒDIGO ##

#Imprimimos las capas de new_model
import pandas as pd
pd.set_option('max_colwidth', -1)
layers = [(layer, layer.name, layer.trainable) for layer in new_model.layers]
print(pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable']))

new_model.summary()

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
new_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# Use generators to save memory
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

H = new_model.fit_generator(aug.flow(x_train, y_train, batch_size=BS),
                            steps_per_epoch=len(x_train) // BS, validation_data=aug.flow(x_test, y_test),
                            epochs=EPOCHS, verbose=2, use_multiprocessing=True)

# save model
# method one
new_model.save('new_model.h5')

# method two
# save model by json and weights
# save json
from keras.models import model_from_json
json_string = new_model.to_json()
with open(r'new_model.json', 'w') as file:
    file.write(json_string)

# save weights
new_model.save_weights('new_model.h5')

# load model
# method one
# model.load('../h5/m_lenet.h5')

# model two
# load model by json and weights
# with open(r'../input/m_lenet.json', 'r') as file:
#     model_json1 = file.read()
#
# model = model_from_json(json_string)
# model.load_weights('../h5/m_weights.h5', by_name=True)
# model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


# plot the iteration process
N = EPOCHS
plt.figure()
plt.plot(np.arange(0,N),H.history['loss'],label='loss')
plt.plot(np.arange(0,N),H.history['accuracy'],label='train_acc')
plt.title('Training Loss and Accuracy on mnist-img classifier')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
plt.savefig('FIGURA.png')

# Calculating loss and accuracy
# train
tr_loss, tr_accurary = new_model.evaluate(x_train, y_train)
# tr_loss = 0.039, tr_accurary = 0.98845
# test
te_loss, te_accurary = new_model.evaluate(x_test, y_test)
# te_loss = 0.042, te_accurary = 0.9861


