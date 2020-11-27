# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("C:\Python37\malaria-tracker\input"))

# Any results you write to the current directory are saved as output.

# %% [markdown]
# > > **Importing the suitable libraries : **

# %% [code]
from PIL import Image
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow.keras
from keras.utils import np_utils
import warnings
#from sklearn.externals
import joblib
warnings.filterwarnings("ignore")
#warnings.FutureWarning("ignore")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout


# %% [code]
os.listdir(os.getcwd())

# %% [code]
os.getcwd()

# %% [code]
data = []
labels= []
data_1=os.listdir("C:\Python37\malaria-tracker\input\Parasitized/")

# %% [code]
for i in data_1:
    try:
        image = cv2.imread("C:\Python37\malaria-tracker\input/Parasitized/"+i)
        image_from_array= Image.fromarray(image , "RGB")
        size_image =image_from_array.resize((50,50))
        #resize45=size_image.rotate(15)
        #resize75 = size_image.rotate(25)
        #blur =cv2.blur(np.array(size_image),(10,10))
        data.append(np.array(size_image))
        labels.append(0)
        #labels.append(0)
        #labels.append(0)
        #labels.append(0)
        
    except AttributeError:
        print("")
Uninfected = os.listdir(r"C:\Python37\malaria-tracker\input\Uninfected/")
for b in Uninfected:
    try :
        image = cv2.imread(r"C:\Python37\malaria-tracker\input\Uninfected/"+b)
        array_image=Image.fromarray(image,"RGB")
        size_image=array_image.resize((50,50))
        resize45= size_image.rotate(15)
        resize75 = size_image.rotate(25)
        #blur =cv2.blur(np.array(size_image),(10,10))
        data.append(np.array(size_image))
        #data.append(np.array(resize45))
        #data.append(np.array(resize75))
        #data.append(np.array(blur))
        #labels.append(1)
        #labels.append(1)
        #labels.append(1)
        labels.append(1)
    except AttributeError:
        print("")

# %% [code]
Cells =np.array(data)
labels =np.array(labels)

# %% [code]
print(labels.shape)
print(Cells.shape)

# %% [code]
#np.save("Cells_data",Cells)
#np.save("labels_data",Cells)

# %% [code]
#Cells =np.load("Cells_data.npy")
#labela =np.load("labels_data.npy")

# %% [code]
s=np.arange(Cells.shape[0])

# %% [code]
np.random.shuffle(s)

# %% [code]
len_data = len(Cells)

# %% [code]
Cells=Cells[s]
labels =labels[s]

# %% [code]
labels =keras.utils.to_categorical(labels)

# %% [code]
model =Sequential()

# %% [code]
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dense(2,activation="softmax"))
model.summary()

# %% [code]
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# %% [code]
Cells=Cells/255

# %% [code]
model.fit(Cells,labels,batch_size=50,epochs=10,verbose=1)

# %% [code]
model.save("my_model111128.h5")

# %% [code]
#!pip install tensorflow==2.0.0-alpha0

import tensorflow as tf

# %% [code]
tf.__version__

# %% [code]
tf.lite.TFLiteConverter

# %% [code]
import tensorflow as tf

# %% [code]
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# %% [code]
converter = tf.lite.TFLiteConverter.from_saved_model("../input/my_model.h5")
tflite_model = converter.convert()

# %% [code]
tf.__version__

# %% [code]


# %% [code]


# %% [code]
joblib.dump(model,"model")

# %% [code]
joblib.load("model")
joblib.dump(model,"model")


# %% [code]
model.save("model111222.h5")

# %% [code]
from keras.models import load_model
model11=load_model("model111.h5")

# %% [code]
model11.predict(Cells[73].reshape(1,50,50,3))

# %% [code]
blur=cv2.blur(Cells[1000].rotate(45),(5,5))

# %% [code]
plt.imshow(blur)

# %% [code]
plt.plot(histroy.history["loss"])#.keys()

# %% [code]
from sklearn.externals import joblib

# %% [code]
joblib.dump(model,"Malaria Cell model")

# %% [code]
from keras.applications.xception import Xception

# %% [code]
model1=Xception()

# %% [code]
modl= keras.applications.vgg16.VGG16()

# %% [code]
modl.summary()

# %% [code]
from keras.applications import VGG16 
vgg_conv = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

# %% [code]
from keras.preprocessing.image import ImageDataGenerator

# %% [code]
train_img=ImageDataGenerator(rescale=1./255,shear_range=0.1,zoom_range=0.2,horizontal_flip=True)

# %% [code]
train_images=train_img.flow_from_directory("C:\Python37\malaria-tracker\input\Parasitized/",target_size=(64,64,3),batch_size=32)
