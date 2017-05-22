
# coding: utf-8

# In[1]:

# Batch Normalization

#Import Modules
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import SGD
import scipy.io
import numpy as np
train = scipy.io.loadmat('train_32x32.mat')
test = scipy.io.loadmat('test_32x32.mat')
x_train = train['X']
y_train = train['y']
x_test = test['X']
y_test = test['y']

y_train[y_train==10]=0
y_test[y_test==10]=0

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 32, 32
X_train_images=np.swapaxes(x_train,2,3)
X_train_images=np.swapaxes(X_train_images,1,2)
X_train_images=np.swapaxes(X_train_images,0,1)

X_test_images=np.swapaxes(x_test,2,3)
X_test_images=np.swapaxes(X_test_images,1,2)
X_test_images=np.swapaxes(X_test_images,0,1)
input_shape = (img_rows, img_cols, 1)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


X_train_images = X_train_images.astype('float32')
X_test_images = X_test_images.astype('float32')
X_train_images /= 255
X_test_images /= 255

from keras.layers import Conv2D, MaxPooling2D, Flatten
input_shape=(32,32,3)
num_classes = 10
from keras.layers import BatchNormalization

# Model Selection
cnn_small_bn = Sequential()
cnn_small_bn.add(Conv2D(32, kernel_size=(3, 3), padding='same',
             activation='relu',
            input_shape=input_shape))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(Conv2D(32, (3, 3),activation='relu',))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Dropout(0.25))
cnn_small_bn.add(Conv2D(64, (3, 3), padding='same'))
cnn_small_bn.add(Activation('relu'))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(Conv2D(64, (3, 3)))
cnn_small_bn.add(Activation('relu'))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(MaxPooling2D(pool_size=(2, 2)))
cnn_small_bn.add(Dropout(0.25))
cnn_small_bn.add(Flatten())
cnn_small_bn.add(Dense(512))
cnn_small_bn.add(Activation('relu'))
cnn_small_bn.add(BatchNormalization())
cnn_small_bn.add(Dropout(0.5))
cnn_small_bn.add(Dense(num_classes))
cnn_small_bn.add(Activation('softmax'))

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

cnn_small_bn.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

#Fitting the Model
history_cnn = cnn_small_bn.fit(X_train_images, y_train,
                      batch_size=128, epochs=2, verbose=1, validation_split=.1,shuffle=True)

# Evaluating the Model
score = cnn_small_bn.evaluate(X_test_images, y_test, verbose=0)

 
# Reporting Loss and Accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:



