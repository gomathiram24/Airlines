
# importing required modules
from keras import layers
from keras import models
import numpy as np 
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import cv2
import glob
from sklearn.model_selection import train_test_split
import os, shutil

# building model
model = Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu'), input_shape=(128, 128, 3)
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(2))
model.summary()

from keras import optimizers

#model compiling
model.compile(loss='sparse_categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

import os, shutil
original_dataset = '/home/ubuntu/Fashion144k_v1/photos'
os.getcwd()
base_dir = '/home/ubuntu/dataset'
os.chdir('/home/ubuntu')
os.mkdir(base_dir)
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)


import keras
import glob
files = [file for file in glob.glob('./Fashion144k_v1/photos/*.jpg')]
import numpy as np
from sklearn.model_selection import train_test_split
(x_train, x_test) = train_test_split(files, test_size=0.25, random_state=42)



with open("./Fashion144k_v1/feat/good.txt", 'rb') as good_files:
    good_file_lines = {int(good_file.decode('utf-8').strip()) - 1 for good_file in good_files.readlines()}


picture = {file: i in good_file_lines for i, file in enumerate(files)}
print(picture[files[5]])


for i, x in enumerate(x_train):
    if i % 100 == 0:
        print(i)
    subdir = "good" if picture[x] else "bad"
    shutil.copyfile(x, os.path.join(train_dir, subdir, x.split("/")[-1]))



for i, x in enumerate(x_test):
    if i % 100 == 0:
        print(i)
    subdir = "good" if picture[x] else "bad"
    shutil.copyfile(x, os.path.join(test_dir, subdir, x.split("/")[-1]))



 from keras.preprocessing.image import ImageDataGenerator

 os.getcwd()
 os.chdir('/home/ubuntu/dataset')



 train_datagen = ImageDataGenerator(rescale=1./255, shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)


 test_datagen = ImageDataGenerator(rescale=1./255, shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)



 train_generator = train_datagen.flow_from_directory(train_dir, target_size=(128, 128), color_mode="rgb", batch_size=128, class_mode="binary")



 test_generator = test_datagen.flow_from_directory(test_dir, target_size=(128, 128), color_mode="rgb", batch_size=128, class_mode="binary")




 model13 = model.fit_generator(train_generator, steps_per_epoch=100, epochs=100, validation_data=test_generator, validation_steps=50)





 
 
