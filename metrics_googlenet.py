# With this script, we obtain the metrics of GoogleNet during a specific training session.

# -*- coding: utf-8 -*-
# @Time    : 2018/12
# @Author  : wengfutian
# @Email   : wengfutian@csu.edu.cn
# Keras for minist classification

# Import libraries
import numpy as np
from keras.utils import np_utils
from keras.optimizers import Adam, Nadam, SGD, Adagrad, RMSprop
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
from mega import Mega
from sklearn import metrics
from keras.models import load_model
import zipfile
from skimage.transform import resize
from numpy import where

mega = Mega()

email = input()
password = input()
m = mega.login(email, password) #ALogin with author's account

details = m.get_user()
print()
print('MEGA Account details:')
print(details)
print()

quota = m.get_quota()
print()
print('MEGA Account quota in MB')
print('Quota: '+str(quota)+' MB')
print()

def get_variable(file):

	file1 = '/Thermy3/'+file
	print(file1[:9])
	print(file1[9:])

	file = m.find(file1)
	m.download(file)
	print(file1[9:]=='X_train2.zip')
	print()
	if file1[9:]=='X_train2.zip' :
		zip_ref = zipfile.ZipFile('/kaggle/working/X_train2.zip', 'r')
		zip_ref.extractall()
		zip_ref.close()
		result=np.load('/kaggle/working/X_train2.npy')
		print()
		print(str(file1[9:])+':')
		print(result)
		print()
	else:
		print
		result=np.load('/kaggle/working/'+file1[9:])
		print()
		print(str(file1[9:])+':')
		print(result)
		print()

	return result

x_train=get_variable('X_train2.zip')

x_test=get_variable('X_test2.npy')

y_train=np.load('/kaggle/working/GoogleNet_Vgg16_TT/Y_train2.npy')
print()
print('y_train:')
print(y_train)
print()

y_test=np.load('/kaggle/working/GoogleNet_Vgg16_TT/Y_test2.npy')
print()
print('y_test:')
print(y_test)
print()

# load minit data
#from keras.datasets import mnist
#(x_train, y_train),(x_test, y_test) = mnist.load_data()

# plot 6 images as gray scale
#matplotlib inline #función mágica
import matplotlib.pyplot as plt

#Función para obtener imagen
def imagen(X,h):
    print(X.shape[0])
    print(X.shape[1])
    image = np.zeros((X.shape[0],X.shape[1]))
    
    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[1]):
          image[i,j] = X[i,j]
    plt.imshow(image, cmap="hot")
    plt.savefig(str(h)+'.png')
    #imgplot = plt.imshow(image)
    #plt.colorbar()
    #imgplot.set_cmap('nipy_spectral')
    plt.show()
    #plt.colorbar()
    
    return image

#Muestra de imágenes de entrenamiento

for i in range(0,4):
    print('Mamas Frontales Ejemplo: '+str(i+1))
    print()
    imagen(x_train[i,:,:,0],i)

# reshape the data to four dimensions, due to the input of model
# reshape to be [samples][width][height][pixels]
#x_train = x_train.reshape(x_train.shape[0], 28, 28, 8).astype('float32')
print('x_train.shape[0]): '+str(x_train.shape[0]))
#x_test = x_test.reshape(x_test.shape[0], 28, 28, 8).astype('float32')

def crop_image_to_list(x):
    list = []
    for k in range(0, x.shape[0]):
        print('Ejemplo: '+str(k))
        for l in range(0, x.shape[3]):
            print('Imagen '+str(l))
            h = x[k,:,:,l]
            #Find up line box L1
            for i in  range(0, x.shape[1]):
                if np.sum(h[i,:])!=0:
                    L1 = i
                    print('Up line box L1: '+str(L1))
                    break
            #Find left line box L2
            for i in  range(0, x.shape[2]):
                if np.sum(h[:,i])!=0:
                    L2 = i
                    print('Left line box L2: '+str(L2))
                    break
            #Find right line box L3
            for i in  range(0, x.shape[2]):
                j = x.shape[2]-1-i
                if np.sum(h[:,j])!=0:
                    L3 = j
                    print('Right line box L3: '+str(L3))
                    break
            #Find down line box L4
            for i in  range(0, x.shape[1]):
                j = x.shape[1]-1-i
                if np.sum(h[j,:])!=0:
                    L4 = j
                    print('Down line box L4: '+str(L4))
                    break
            list.append(h[L1:L4+1,L2:L3+1])
            plt.imshow(h[L1:L4+1,L2:L3+1], cmap="hot")
            if k==0 and l==0:
                plt.savefig(str(k)+'_1'+'.png')
                print(k)
                print(l)
                print()
            if k==1 and l==0:
                plt.savefig(str(k)+'_1'+'.png')
                print(k)
                print(l)
            if k==2 and l==0:
                plt.savefig(str(k)+'_1'+'.png')
                print(k)
                print(l)
            if k==3 and l==0:
                plt.savefig(str(k)+'_1'+'.png')
                print(k)
                print(l)
            plt.show()
    return list

#Eliminando mayoría de ceros y almacenando en lista x_raw
x_raw=crop_image_to_list(x_train)

filas = np.zeros((len(x_raw)))
columnas = np.zeros((len(x_raw)))

for i in range(0, len(x_raw)):
    filas[i] = x_raw[i].shape[0]
    columnas[i] = x_raw[i].shape[1]

print(i)

def list_resize_to_np(list, examples = 0, pictures = 'False'):
    x=np.zeros((examples, int(np.max(filas)), int(np.max(columnas)), 8))
    for i in range(0, examples):
        print(i)
        j=i*8 #De 8 en 8 sobre los elementos de list
        temp = list[j+0]
        temp = temp.copy()
        x[i,:,:,0] = resize(temp, (int(np.max(filas)), int(np.max(columnas))))
        if pictures == 'True':
        	if i==0 or i==1 or i==2 or i==3:
        		plt.imshow(x[i,:,:,0], cmap="hot")
        		plt.savefig(str(i)+'_2'+'.png')
        temp = list[j+1]
        temp = temp.copy()
        x[i,:,:,1] = resize(temp, (int(np.max(filas)), int(np.max(columnas))))
        temp = list[j+2]
        temp = temp.copy()
        x[i,:,:,2] = resize(temp, (int(np.max(filas)), int(np.max(columnas))))
        temp = list[j+3]
        print(j+3)
        temp = temp.copy()
        x[i,:,:,3] = resize(temp, (int(np.max(filas)), int(np.max(columnas))))
        temp = list[j+4]
        temp = temp.copy()
        x[i,:,:,4] = resize(temp, (int(np.max(filas)), int(np.max(columnas))))
        temp = list[j+5]
        temp = temp.copy()
        x[i,:,:,5] = resize(temp, (int(np.max(filas)), int(np.max(columnas))))
        temp = list[j+6]
        temp = temp.copy()
        x[i,:,:,6] = resize(temp, (int(np.max(filas)), int(np.max(columnas))))
        temp = list[j+7]
        temp = temp.copy()
        x[i,:,:,7] = resize(temp, (int(np.max(filas)), int(np.max(columnas))))
    return x

#Almacenando imágenes recortadas y extrapoladas
x_train = list_resize_to_np(x_raw, examples = x_train.shape[0], pictures = 'True')

#Función para Tranformación Lineal de entradas X
#Cargamos datos necesarios

MinMax = np.load('/kaggle/working/GoogleNet_Vgg16_TT/MinMaxGlobales.npy')
TminBas = MinMax[0]
TminFunc = MinMax[1]
TmaxBas = MinMax[2]
TmaxFunc = MinMax[3]

#Definiendo límites de la tranformación
L1 = 0
L2 = 255 #tranformación a imagen digital de 8 bits


#Convirtiendo x_train Basales (TRANSFORMACIÓN LINEAL)

T1 = TminBas - 0.5
T2 = TmaxBas + 0.5

x_train[:,:,:,0:4]=((L2-L1)/(T2-T1))*(x_train[:,:,:,0:4]-T1)+L1

#Convirtiendo X_train Funcionales (TRANSFORMACIÓN LINEAL)

T1 = TminFunc - 0.5
T2 = TmaxFunc + 0.5

x_train[:,:,:,4:]=((L2-L1)/(T2-T1))*(x_train[:,:,:,4:]-T1)+L1


#De forma similar con x_test
#Eliminando mayoría de ceros y almacenando en lista x_raw2
x_raw2=crop_image_to_list(x_test)
#Almacenando imágenes recortadas y extrapoladas
#El mismo resize que en el entrenamiento
x_test = list_resize_to_np(x_raw2, examples = x_test.shape[0])

#Convirtiendo x_test Basales (TRANSFORMACIÓN LINEAL)

T1 = TminBas - 0.5
T2 = TmaxBas + 0.5

x_test[:,:,:,0:4]=((L2-L1)/(T2-T1))*(x_test[:,:,:,0:4]-T1)+L1

#Convirtiendo X_test Funcionales (TRANSFORMACIÓN LINEAL)

T1 = TminFunc - 0.5
T2 = TmaxFunc + 0.5

x_test[:,:,:,4:]=((L2-L1)/(T2-T1))*(x_test[:,:,:,4:]-T1)+L1


#Estableciendo cualquier valor negativo a 0
x_train[where(x_train <= 0)] = 0
x_test[where(x_test <= 0)] = 0


#Aquí eliminamos o reasignamos canales a conveniencia para pruebas
#x_train = x_train[:,:,:,:2]
#x_test = x_test[:,:,:,:2]  

### SUMANDO CANALES 0 Y 1 ###
#Los 8 canales a 4 canales

def sum_channels(X, pictures=False):
    x_temp = np.zeros((X.shape[0], X.shape[1], 2*X.shape[2], 2))
    for i in range(0, X.shape[0]):
        x_temp[i,:,0:X.shape[2],0] = X[i,:,:,0]
        x_temp[i,:,X.shape[2]:,0] = X[i,:,:,1]
        #x_temp[i,:,0:X.shape[2],1] = X[i,:,:,2]
        #x_temp[i,:,X.shape[2]:,1] = X[i,:,:,3]
        x_temp[i,:,0:X.shape[2],1] = X[i,:,:,2]
        x_temp[i,:,X.shape[2]:,1] = X[i,:,:,3]
        #x_temp[i,:,0:X.shape[2],3] = X[i,:,:,6]
        #x_temp[i,:,X.shape[2]:,3] = X[i,:,:,7]
        if pictures == 'True':
            if i == 0 or i == 1 or i == 2 or i == 3:
                plt.imshow(x[i,:,:,0], cmap="hot")
                plt.savefig(str(i)+'_3'+'.png')
    return x_temp

x_train = sum_channels(x_train, pictures=True)
print()
print('Valor Máximo de x_train:')
print(np.max(x_train))
print('Valor Mínimo de x_test:')
print(np.min(x_train))

x_test = sum_channels(x_test)
print()
print('Valor Máximo de x_test:')
print(np.max(x_test))
print('Valor Mínimo de x_test:')
print(np.min(x_test))

### TERMINA SUMA DE CANALES ###


# normalization
x_train = x_train / 255.0
print()
print('Valor Máximo de x_train:')
print(np.max(x_train))
print('Valor Mínimo de x_test:')
print(np.min(x_train))

x_test = x_test / 255.0
print()
print('Valor Máximo de x_test:')
print(np.max(x_test))
print('Valor Mínimo de x_test:')
print(np.min(x_test))

#Reestructuring x_train and x_test
#for i in range(0,len(x_train.shape[0])):
#	for j in range(0, 8):


# one-hot
y_train = np_utils.to_categorical(y_train)
print()
print('y_train:')
print(y_train)
y_test = np_utils.to_categorical(y_test)
print()
print('y_test:')
print(y_test)
print()

#Función que Convierte un arreglo one_hot a uno flatt con las clases corespondientes
def convert_to_one(Y):
  temp = np.array([[0]])
  Y = np.asarray(Y, dtype=np.float64)
  for i in range(0, Y.shape[0]):
    temp1 = int(np.argmax(Y[i,:]))
    print('Valor por valor del vector')
    print(Y[i,:])
    print(np.nonzero(Y[i,:])[0][0])
    print(np.argmax(Y[i,:]))
    temp = np.append(temp, [[temp1]], axis=0)
  print()
  print('Este es el vector final')
  print(temp)
  print()
  Y = np.delete(temp, 0, 0)
  return Y

#Load googlenet architecture
#Google Colab '/content/GoogleNet_Vgg16_thermy/googlenet.json'
#Kaggle
file = m.find('/Thermy3/googlenet.json')
m.download(file)
with open('/kaggle/working/googlenet.json','r') as f:
    model_json = json.load(f)

#Load googlenet model weights
googlenet_model = model_from_json(json.dumps(model_json))

#Import googlenet model with 'he_uniform' intialization
#Kaggle
file = m.find('/Thermy3/googlenet.h5')
m.download(file)
googlenet_model.load_weights("/kaggle/working/googlenet.h5")
#Google Colab
#googlenet_model.load_weights("/content/GoogleNet_Vgg16/googlenet.h5")

googlenet_model.summary()

model = googlenet_model

EPOCHS = 1000
INIT_LR = 1e-3

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
#Parámetros de Training

p = model.predict(x_train)

print('Probabilidad por clase: ')
print(p)

p1 = model.evaluate(x = x_train, y = y_train)

y_train_ = convert_to_one(y_train)
print('Imprimmos Y_train_')
print(y_train_)

p_ = convert_to_one(p)
print('Imprimiendo las predicciones p_')
print(p_)

matrix1 = metrics.confusion_matrix(y_train_, p_)

tn1, fp1, fn1, tp1 = matrix1.ravel()

acc = (tn1 + tp1)/(tn1 + fp1  + fn1 + tp1)

s = tn1/(tn1 + fp1) #Especificidad (Specifcity)

r = tp1/(tp1 + fn1) #Sensibilidad (Recall)

pre = tp1/(tp1 + fp1) #Precisión

f1 =  2 * pre*r/(pre + r) #F1 Score

print("Numero de positivos: "+str(np.sum(y_train_)))
print("preds.shape: "+str(p.shape))
print("Y_train.shape: "+str(y_train.shape))
print ("Loss = " + str(p1[0]))
print ("Train Accuracy (Keras) = " + str(p1[1]))
print("Train Confusion Matrix" + str(matrix1))
print("tn1: " + str(tn1))
print("fp1: " + str(fp1))
print("fn1: " + str(fn1))
print("tp1: " + str(tp1))
print()
print('Train Accuracy (Scikit-Learn) = ' + str(acc))
print('Specifcity = ' + str(s))
print('Recall = ' + str(r))
print('Precision = ' + str(pre))
print('F1 Score = ' + str(f1))


#Parámetros de Validación

pred = model.predict(x_test)

print('Probabilidad por clase: ')
print(pred)

p2 = model.evaluate(x = x_test, y = y_test)

y_test_ = convert_to_one(y_test)
print('Imprimmos Y_test_')
print(y_test_)

p_2 = convert_to_one(pred)
print('Imprimiendo las predicciones p_2')
print(p_2)

matrix2 = metrics.confusion_matrix(y_test_, p_2)

tn2, fp2, fn2, tp2 = matrix2.ravel()

acc = (tn2 + tp2)/(tn2 + fp2  + fn2 + tp2)

s = tn2/(tn2 + fp2) #Especificidad (Specifcity)

r = tp2/(tp2 + fn2) #Sensibilidad (Recall)

pre = tp2/(tp2 + fp2) #Precisión

f1 =  2 * pre*r/(pre + r) #F1 Score

print("Numero de positivos: "+str(np.sum(y_test_)))
print("preds.shape: "+str(pred.shape))
print("Y_test.shape: "+str(y_test.shape))
print ("Loss = " + str(p2[0]))
print ("Test Accuracy (Keras) = " + str(p2[1]))
print("Test Confusion Matrix" + str(matrix2))
print("tn2: " + str(tn2))
print("fp2: " + str(fp2))
print("fn2: " + str(fn2))
print("tp2: " + str(tp2))
print()
print('Test Accuracy (Scikit-Learn) = ' + str(acc))
print('Specifcity = ' + str(s))
print('Recall = ' + str(r))
print('Precision = ' + str(pre))
print('F1 Score = ' + str(f1))
