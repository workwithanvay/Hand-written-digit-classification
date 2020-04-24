#Importing the libraries
import pandas as pd
import numpy as np
np.random.seed(1212)
import keras
from keras.models import Model
from keras.layers import *
from keras import optimizers
from keras.layers import Dense
from keras.layers import Dropout

#Reading train and test set
training_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

#Splitting training and validation set
X = training_set.iloc[:, 1:785]
Y = training_set.iloc[:, 0]
X_test =test_set.iloc[:, 0:784]


from sklearn.model_selection import train_test_split
X_train, X_cv, y_train, y_cv = train_test_split(X,Y, 
                                                test_size = 0.25,
                                                random_state = 1212)
'''
X_train = X_train.as_matrix().reshape(31500, 784) #(33600, 784)
X_cv = X_cv.as_matrix().reshape(10,500, 784) #(8400, 784)

X_test = X_test.as_matrix().reshape(28000, 784)'''

# Feature Normalization 
X_train = X_train.astype('float32'); X_cv= X_cv.astype('float32'); X_test = X_test.astype('float32')
X_train /= 255; X_cv /= 255; X_test /= 255

# Convert labels to One Hot Encoded

num_digits = 10
y_train = keras.utils.to_categorical(y_train, num_digits)
y_cv = keras.utils.to_categorical(y_cv, num_digits)

#Model
# Input Parameters
n_input = 784 
n_hidden_1 = 300
n_hidden_2 = 100
n_hidden_3 = 100
n_hidden_4 = 200
num_digits = 10

Inp = Input(shape=(784,))
x = Dense(n_hidden_1, activation='relu', name = "Hidden_Layer_1")(Inp)
x = Dropout(0.3)(x)
x = Dense(n_hidden_2, activation='relu', name = "Hidden_Layer_2")(x)
x = Dropout(0.3)(x)
x = Dense(n_hidden_3, activation='relu', name = "Hidden_Layer_3")(x)
x = Dropout(0.3)(x)
x = Dense(n_hidden_4, activation='relu', name = "Hidden_Layer_4")(x)
output = Dense(num_digits, activation='softmax', name = "Output_Layer")(x)

# Our model would have '6' layers - input layer, 4 hidden layer and 1 output layer
model = Model(Inp, output)
model_summary=model.summary() # We have 297,910 parameters to estimate

# Insert Hyperparameters
learning_rate = 0.1
training_epochs = 20
batch_size = 100

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size = batch_size,
                    epochs = training_epochs,
                    validation_data=(X_cv, y_cv))
