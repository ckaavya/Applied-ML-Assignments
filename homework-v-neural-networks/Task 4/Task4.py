
# coding: utf-8

# In[ ]:

# Import Modules
from keras import applications
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import os
import numpy as np
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


print('made model')
# build the VGG16 network
model = applications.VGG16(include_top=False,
                           weights='imagenet')

model.summary()

vgg_weights, vgg_biases = model.layers[1].get_weights()



p = open('/rigel/home/as5196/scripts/datasets/list.txt', 'r')

imgid = []
y = []

for line in p:
    if line[0][0] != '#':
        l = line.strip().split()
        if len(l) == 4:
            imgid.append(l[0] +'.jpg')
            y.append(l[1] + '_' + l[2] + '_' + l[3])
p.close()
print('made list of imgids')




images = [image.load_img(os.path.join("/rigel/home/as5196/scripts/datasets/images/",os.path.basename(i)), target_size=(500, 375)) for i in imgid]
print('loaded images')


X = np.array([image.img_to_array(img) for img in images])

print(X.shape)


X_pre = preprocess_input(X)
features = model.predict(X_pre)

print(features.shape)
print('done features')

features_ = features.reshape(X.shape[0], -1)


y_ = y
y_ = np.array(y_)

X_train, X_test, y_train, y_test = train_test_split(features_, y_)
print('Running LR')

lr = LogisticRegression(C = 1).fit(X_train, y_train)

print(lr.score(X_train, y_train))
print(lr.score(X_test, y_test))                        
[5:54 AM, 4/29/2017] Amla, Columbia: final                        
[5:54 AM, 4/29/2017] Amla, Columbia: #importing packages
from keras import applications
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import os
import numpy as np
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# build the VGG16 network
model = applications.VGG16(include_top=False,
                           weights='imagenet')

model.summary()

vgg_weights, vgg_biases = model.layers[1].get_weights()

#read list.txt for all image classifications
p = open('/rigel/home/as5196/scripts/datasets/list.txt', 'r')

imgid = []
y = []

for line in p:
    if line[0][0] != '#':
        l = line.strip().split()
        if len(l) == 4:
            imgid.append(l[0] +'.jpg')
            y.append(int(l[1]))
p.close()

#load images
images = [image.load_img(os.path.join("/rigel/home/as5196/scripts/datasets/images/",os.path.basename(i)), target_size=(500, 375)) for i in imgid]


#convert images to numpy arrays
X = np.array([image.img_to_array(img) for img in images])
print(X.shape)

#feature extraction
X_pre = preprocess_input(X)
features = model.predict(X_pre)
print(features.shape)

features_ = features.reshape(X.shape[0], -1)

y_ = y
y_ = np.array(y_)

#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_, y_)

#building linear model for classification
lr = LogisticRegression(C = 1).fit(X_train, y_train)

print('Training accuracy:')
print(lr.score(X_train, y_train))
print('Testing accuracy:')
print(lr.score(X_test, y_test))

