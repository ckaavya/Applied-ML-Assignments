
# coding: utf-8

# In[8]:

# Import modules
import numpy as np

# Setting seed value
np.random.seed(100)
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

# Loading the iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X, y = shuffle(X, y)

# Convert y to categorical
y = keras.utils.to_categorical(y, num_classes=3)

# Splitting data to train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Model Selection
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Model Summary
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fitting the model
model.fit(X_train, y_train,
          epochs=20,
          batch_size=128)

# Model Evaluation
score = model.evaluate(X_test, y_test, batch_size=128)

# Reporting test data loss and accuracy
print("Test loss: {:.3f}%".format(score[0]*100))
print("Test Accuracy: {:.3f}%".format(score[1]*100))


# In[ ]:



