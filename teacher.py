
# coding: utf-8

# In[38]:



# coding: utf-8

# In[3]:


import os
import time
from collections import defaultdict
import cv2
import random
import keras
from keras.layers import Activation, Input, Embedding, LSTM, Dense, Lambda, GaussianNoise, concatenate
from keras.models import Model
import numpy as np
import json
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Activation
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.constraints import max_norm
from keras.layers import MaxPooling2D, Dropout, Dense, Flatten, Activation, Conv2D
from keras.models import Sequential
from keras.losses import categorical_crossentropy as logloss
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.layers.normalization import BatchNormalization
from keras_preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
np.random.seed(1000)
import pandas as pd
from numpy import argmax
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.layers import AveragePooling2D
from keras.applications import DenseNet169
from keras.applications.mobilenet import preprocess_input


path_df = pd.read_csv('train_full.csv')


train = path_df.head(int(len(path_df)*(80/100)))

test = path_df.tail(int(len(path_df)*(20/100)))





# In[ ]:



label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train["Label"])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y_train = onehot_encoder.fit_transform(integer_encoded)

label_encoder_test = LabelEncoder()
integer_encoded_test = label_encoder_test.fit_transform(test["Label"])
onehot_encoder_test = OneHotEncoder(sparse=False)
integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)
Y_test = onehot_encoder_test.fit_transform(integer_encoded_test)


# In[1]:

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Teacher model
filepath = "mura-densenet169_full15.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

input_shape = (224, 224, 3) # Input shape of each image

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
base_model = DenseNet169(weights="imagenet",include_top=False, input_shape = (224, 224, 3)) #imports the mobilenet model and discards the last 1000 neuron layer.

x = AveragePooling2D(pool_size=(3,3), name='avg_pool')(base_model.output)
x = Flatten()(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = BatchNormalization()(x)
x = Dropout(0.5, name='dropout_fc2')(x)
preds = Dense(14, activation="softmax", name="preds")(x)

teacher=Model(inputs=base_model.input,outputs=preds)

teacher.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])

print(teacher.summary())


# In[ ]:


# In[ ]:

class My_Generator(keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size, shuffle):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([np.around(np.array(cv2.resize(cv2.imread(file_name), dsize=(224, 224), interpolation=cv2.INTER_CUBIC) / 255), decimals=2) for file_name in batch_x]), np.array(batch_y)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.image_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


# In[52]:


batch_size= 16
num_training_samples = len(train)
num_validation_samples = len(test)
num_epochs = 50
shuffle = True

my_training_batch_generator = My_Generator(np.array(train['Path']), Y_train, batch_size, shuffle)
my_validation_batch_generator = My_Generator(np.array(test['Path']), Y_test, batch_size, shuffle)

hist = teacher.fit_generator(generator=my_training_batch_generator,
                                          steps_per_epoch= int(num_training_samples // batch_size),
                                          epochs=num_epochs,
                                          verbose=1,
                                          validation_data= my_validation_batch_generator,
                                          validation_steps= int(num_validation_samples // batch_size),
                                          use_multiprocessing=False,
                                          shuffle=True,
                                          callbacks = [checkpoint])


# In[ ]:


# In[ ]:

with open('mura-densenet169-hist_full15.json', 'w') as f:
    json.dump(hist.history, f)

# serialize model to JSON
model_json = teacher.to_json()
with open("mura-densenet169_full15.json", "w") as json_file:
    json_file.write(model_json)



# In[4]:


# load json and create model
json_file = open('mura-densenet169_full15.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("mura-densenet169_full15.h5")
print("Loaded model from disk")

