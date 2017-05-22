
# coding: utf-8

# In[9]:

# Import modules
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from keras.datasets import mnist
import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.utils import shuffle
import pandas as pd

# Load mnist data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Getting the training and test set
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

num_classes = 10


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Model Selection
model = Sequential([
    Dense(128, input_shape=(784,)),
    Activation('relu'),
    Dropout(0.5),
    Dense(64),
    Activation('relu'),
    Dense(10),
    Dropout(0.5),
    Activation('softmax'),
])

model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])

model.summary()

# Fitting the model
history=model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1,validation_split=.1)

# Model Evaluation
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: {:.3f}%".format(score[0]*100))
print("Test Accuracy: {:.3f}%".format(score[1]*100))

# Reporting test data loss and accuracy
df = pd.DataFrame(history.history)
df[['acc', 'val_acc']].plot()
plt.ylabel("accuracy")
df[['loss', 'val_loss']].plot(linestyle='--', ax=plt.twinx())
plt.ylabel("loss")

plt.savefig("Part2.png")


# In[ ]:



