# importing dependencies
from sklearn.model_selection import train_test_split as tts
import pickle
from keras.utils import normalize
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from termcolor import *
import colorama

colorama.init()

# importing data
model_dictionary = pickle.load(open('model_dictionary', 'rb'))
features = pickle.load(open('features', 'rb'))
labels = pickle.load(open('labels', 'rb'))

# Make training and testing data:
x_train, x_test, y_train, y_test = tts(features, labels, test_size = 0.5)

# normalize data
x_train, x_test = normalize((x_train, x_test), axis = 1)
y_train, y_test = np.array((y_train, y_test))

# Make model
model = Sequential()
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128 , activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

# compile the model
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
			  metrics = ['accuracy'])

# train the model
model.fit(x_train, y_train, epochs = 2)

# evaluate the model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(f'Testing loss: {val_loss}, Testing accuracy: {val_acc}')

# saving model
model.save('classifier_dl')
