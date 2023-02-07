# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 21:38:50 2023

@author: ThinkPad
"""

import tensorflow as tf
import os

# Remove dodgy images -------------------------------
import cv2
import imghdr
image_dir = "data_image"
image_extension = ['jpeg','jpg','bmp','png']

for image_dir_list in os.listdir(image_dir):
    #print(image_dir_list)
    for images in os.listdir(os.path.join(image_dir, image_dir_list)):
        image_path = os.path.join(image_dir, image_dir_list,images)
        #print(image_path)
        try:
        
            image_inside_data = cv2.imread(image_path)
            img_ext = imghdr.what(image_path)
            if img_ext not in image_extension:
                print("Images extension does not exist in extension list...{}".format(image_path))
                os.remove(image_path)
        except  Exception as e:
            print("There is a problem with this image...{}".format(image_path))

#----------------------------------------------------
# Load Data -----------------------------------------
import numpy as np
from matplotlib import pyplot as plt

data = tf.keras.utils.image_dataset_from_directory("data_image")
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

fig, ax = plt.subplots(ncols=4, figsize=(20,20) )
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])
#----------------------------------------------------
# Scale Data -----------------------------------------
data = data.map(lambda x,y: (x/255, y))
data.as_numpy_iterator().next()
#----------------------------------------------------
# Split Data -----------------------------------------
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)
train_size
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)
#----------------------------------------------------
# Build Deep Learning Model -----------------------------------------
train
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

model = Sequential()
model.add(Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), 1, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()
#----------------------------------------------------
# Train Data -----------------------------------------
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=20, validation_data=val, callbacks=[tensorboard_callback])
#----------------------------------------------------
# Plot Performance -----------------------------------------
fig = plt.figure()
plt.plot(hist.history['loss'], color='teal', label='loss')
plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
fig.suptitle('Loss', fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
fig.suptitle('Accuracy', fontsize=20)
plt.legend(loc="upper left")
plt.show()
#----------------------------------------------------
# Evaluate -----------------------------------------
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as_numpy_iterator(): 
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(pre.result(), re.result(), acc.result())
#----------------------------------------------------
# Test -----------------------------------------
import cv2

img = cv2.imread('test1.jpg')
plt.imshow(img)
plt.show()

resize = tf.image.resize(img, (256,256))
plt.imshow(resize.numpy().astype(int))
plt.show()

yhat = model.predict(np.expand_dims(resize/255, 0))

yhat

if yhat > 0.5: 
    print(f'Predicted class is Sad')
else:
    print(f'Predicted class is Happy')
#----------------------------------------------------
# Save the model -----------------------------------------
from tensorflow.keras.models import load_model

model.save(os.path.join('models','imageclassifier.h5'))

new_model = load_model(os.path.join('models/','imageclassifier.h5'))

new_model.predict(np.expand_dims(resize/255, 0))


#----------------------------------------------------





