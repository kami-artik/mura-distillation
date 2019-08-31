
# coding: utf-8

# In[2]:



# coding: utf-8

# In[2]:


import os
import time
from collections import defaultdict
import cv2
import random

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
from keras import Model


# In[9]:

print('Temp: 4')

data = pd.read_csv('test_full.csv')


# In[ ]:





# In[2]:


temp = 4


# In[3]:



# load json and create model
json_file = open('/homedtic/kartik/mura_student_fullt4_new.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects = {"temp": temp})
# load weights into new model
loaded_model.load_weights("/homedtic/kartik/mura_student_fullt4_new.h5")
print("Loaded model from disk")
json_file = open('/homedtic/kartik/mura_student-hist_fullt4_new.json', 'r')


# In[12]:


# In[14]:


test_data = []
for image in data["Path"]:
    test_data.append(cv2.resize(cv2.imread(image), dsize=(224, 224), interpolation=cv2.INTER_CUBIC) / 255) #Resize and normalize
test_data = np.array(test_data)    


# In[15]:


label_encoder_test = LabelEncoder()
integer_encoded_test = label_encoder_test.fit_transform(data["Label"])
onehot_encoder_test = OneHotEncoder(sparse=False)
integer_encoded_test = integer_encoded_test.reshape(len(integer_encoded_test), 1)
Y_test = onehot_encoder_test.fit_transform(integer_encoded_test)


# In[18]:


#Collect the logits from the previous layer output and store it in a different model
teacher_WO_Softmax = Model(loaded_model.input, loaded_model.get_layer('dense_4').output)

# Define a manual softmax function
def softmax(x):
    return np.exp(x)/(np.exp(x).sum())


teacher_test_logits = teacher_WO_Softmax.predict(test_data) # This model directly gives the logits ( see the teacher_WO_softmax model above)


Y_test_soft = softmax(teacher_test_logits / temp)

Y_test_new =  np.concatenate([Y_test, Y_test_soft], axis =1)


# In[17]:


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

def specificity(y_true, y_pred):
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity


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

loaded_model.compile(
    #optimizer=optimizers.SGD(lr=1e-1, momentum=0.9, nesterov=True),
    optimizer='adam',
    loss=lambda y_true, y_pred: knowledge_distillation_loss(y_true, y_pred, 0.1),
    #loss='categorical_crossentropy',
    metrics=[acc, f1_m,precision_m, recall_m,specificity]) 


# In[18]:

scores = loaded_model.evaluate(test_data, Y_test_new, verbose=0)
scores_wrist = loaded_model.evaluate(test_data[0:658], Y_test_new[0:658], verbose=0)
scores_forearm = loaded_model.evaluate(test_data[659:959], Y_test_new[659:959], verbose=0)
scores_hand = loaded_model.evaluate(test_data[960:1419], Y_test_new[960:1419], verbose=0)
scores_humerus = loaded_model.evaluate(test_data[1420:1708], Y_test_new[1420:1708], verbose=0)
scores_shoulder = loaded_model.evaluate(test_data[1709:2270], Y_test_new[1709:2270], verbose=0)
scores_elbow = loaded_model.evaluate(test_data[2271:2735], Y_test_new[2271:2735], verbose=0)
scores_finger = loaded_model.evaluate(test_data[2736:3196], Y_test_new[2736:3196], verbose=0)

# In[19]:
print("OVERALL")
print("loss =", scores[0])
print("accuracy =", scores[1])
print("f1 =", scores[2])
print("precision =", scores[3])
print("recall =", scores[4])
print("specificity =", scores[5])
print("********************************************")
print("WRIST")
print("loss =", scores_wrist[0])
print("accuracy =", scores_wrist[1])
print("f1 =", scores_wrist[2])
print("precision =", scores_wrist[3])
print("recall =", scores_wrist[4])
print("specificity =", scores_wrist[5])
print("********************************************")
print("ELBOW")
print("loss =", scores_elbow[0])
print("accuracy =", scores_elbow[1])
print("f1 =", scores_elbow[2])
print("precision =", scores_elbow[3])
print("recall =", scores_elbow[4])
print("specificity =", scores_elbow[5])
print("********************************************")
print("FINGER")
print("loss =", scores_finger[0])
print("accuracy =", scores_finger[1])
print("f1 =", scores_finger[2])
print("precision =", scores_finger[3])
print("recall =", scores_finger[4])
print("specificity =", scores_finger[5])
print("********************************************")
print("HAND")
print("loss =", scores_hand[0])
print("accuracy =", scores_hand[1])
print("f1 =", scores_hand[2])
print("precision =", scores_hand[3])
print("recall =", scores_hand[4])
print("specificity =", scores_hand[5])
print("********************************************")
print("HUMERUS")
print("loss =", scores_humerus[0])
print("accuracy =", scores_humerus[1])
print("f1 =", scores_humerus[2])
print("precision =", scores_humerus[3])
print("recall =", scores_humerus[4])
print("specificity =", scores_humerus[5])
print("********************************************")
print("FOREARM")
print("loss =", scores_forearm[0])
print("accuracy =", scores_forearm[1])
print("f1 =", scores_forearm[2])
print("precision =", scores_forearm[3])
print("recall =", scores_forearm[4])
print("specificity =", scores_forearm[5])
print("********************************************")
print("SHOULDER")
print("loss =", scores_shoulder[0])
print("accuracy =", scores_shoulder[1])
print("f1 =", scores_shoulder[2])
print("precision =", scores_shoulder[3])
print("recall =", scores_shoulder[4])
print("specificity =", scores_shoulder[5])
print("********************************************")

