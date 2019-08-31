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
import ssl
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
from keras import models
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
from keras.layers import GlobalAveragePooling2D
from keras.applications import VGG19, MobileNet
from keras.applications.mobilenet import preprocess_input
from keras import layers
from keras import optimizers

# The csv file contains image paths and labels
path_df = pd.read_csv('train_full.csv')


train = path_df.head(int(len(path_df)*(80/100)))

valid = path_df.tail(int(len(path_df)*(20/100)))


batch_size = 16
num_epochs = 50


train_data = []
for image in train["Path"]:
    train_data.append(cv2.resize(cv2.imread(image), dsize=(224, 224), interpolation=cv2.INTER_CUBIC) / 255) #Resize and normalize

    
valid_data = []
for image in valid["Path"]:
    valid_data.append(cv2.resize(cv2.imread(image), dsize=(224, 224), interpolation=cv2.INTER_CUBIC) / 255) #Resize and normalize

valid_data = np.array(valid_data)
train_data = np.array(train_data)    
    
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(train["Label"])
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
Y_train = onehot_encoder.fit_transform(integer_encoded)

label_encoder_valid = LabelEncoder()
integer_encoded_valid = label_encoder_valid.fit_transform(valid["Label"])
onehot_encoder_valid = OneHotEncoder(sparse=False)
integer_encoded_valid = integer_encoded_valid.reshape(len(integer_encoded_valid), 1)
Y_valid = onehot_encoder_valid.fit_transform(integer_encoded_valid)


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


# Teacher model
filepath = "train_with_distillation.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

input_shape = (224, 224, 3) # Input shape of each image

ssl._create_default_https_context = ssl._create_unverified_context
base_model = MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
x=Dropout(0.75)(x)#we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dropout(0.75)(x)
x=Dense(512,activation='relu')(x) #dense layer 3
x=Dense(14)(x)
preds=Activation('softmax')(x) #final layer with softmax activation

student=Model(inputs=base_model.input,outputs=preds)

student.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',f1_m,precision_m, recall_m])

print(student.summary())


# load json and create model
json_file = open('teacher.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
teacher = model_from_json(loaded_model_json)
# load weights into new model
teacher.load_weights("teacher.h5")
print("Loaded model from disk")


# Raise the temperature of teacher model and gather the soft targets

# Set a tempature value
temp = 5

#Collect the logits from the previous layer output and store it in a different model
teacher_WO_Softmax = Model(teacher.input, teacher.get_layer('preds').output)


# Define a manual softmax function
def softmax(x):
    return np.exp(x)/(np.exp(x).sum())


teacher_train_logits = teacher_WO_Softmax.predict(train_data)
teacher_valid_logits = teacher_WO_Softmax.predict(valid_data) # This model directly gives the logits ( see the teacher_WO_softmax model above)

# Perform a manual softmax at raised temperature
train_logits_T = teacher_train_logits/ temp
valid_logits_T = teacher_valid_logits / temp 

Y_train_soft = softmax(train_logits_T)
Y_valid_soft = softmax(valid_logits_T)

# Concatenate so that this becomes a 10 + 10 dimensional vector
Y_train_new = np.concatenate([Y_train, Y_train_soft], axis=1)
Y_valid_new =  np.concatenate([Y_valid, Y_valid_soft], axis =1)


# Remove the softmax layer from the student network
student.layers.pop()

# Now collect the logits from the last layer
logits = student.layers[-1].output # This is going to be a tensor. And hence it needs to pass through a Activation layer
probs = Activation('softmax')(logits)

# softed probabilities at raised temperature
logits_T = Lambda(lambda x: x / temp)(logits)
probs_T = Activation('softmax')(logits_T)

output = concatenate([probs, probs_T])

# This is our new student model
student = Model(student.input, output)

student.summary()


nb_classes = 14

# This will be a teacher trained student model. 
# --> This uses a knowledge distillation loss function

# Declare knowledge distillation loss
def knowledge_distillation_loss(y_true, y_pred, alpha):

    # Extract the one-hot encoded values and the softs separately so that we can create two objective functions
    y_true, y_true_softs = y_true[: , :nb_classes], y_true[: , nb_classes:]
   
    y_pred, y_pred_softs = y_pred[: , :nb_classes], y_pred[: , nb_classes:]
    
    loss = alpha*logloss(y_true,y_pred) + logloss(y_true_softs, y_pred_softs)
    
    return loss

# For testing use regular output probabilities - without temperature
def acc(y_true, y_pred):
    y_true = y_true[:, :nb_classes]
    y_pred = y_pred[:, :nb_classes]
    return categorical_accuracy(y_true, y_pred)

student.compile(
    #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
    optimizer='adam',
    loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, 0.1),
    #loss='categorical_crossentropy',
    metrics=[acc, f1_m,precision_m, recall_m]) 



num_training_samples = len(train)
num_validation_samples = len(valid)
shuffle = True

my_training_batch_generator = My_Generator(np.array(train['Path']), Y_train_new, batch_size, shuffle)
my_validation_batch_generator = My_Generator(np.array(valid['Path']), Y_valid_new, batch_size, shuffle)

hist_student = student.fit_generator(generator=my_training_batch_generator,
                                          steps_per_epoch= int(num_training_samples // batch_size),
                                          epochs=num_epochs,
                                          verbose=1,
                                          validation_data= my_validation_batch_generator,
                                          validation_steps= int(num_validation_samples // batch_size),
                                          use_multiprocessing=False,
                                          shuffle=True,
                                          callbacks = [checkpoint]) 


with open('student_hist.json', 'w') as f:
    json.dump(hist_student.history, f)

# serialize model to JSON
model_json = student.to_json()
with open("student.json", "w") as json_file:
    json_file.write(model_json)


# load json and create model
json_file = open('student.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={'temp': temp})
# load weights into new model
loaded_model.load_weights("student.h5")
print("Loaded model from disk")

