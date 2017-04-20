import os
import io
import math
import codecs
import numpy as np
import string
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import tensorflow as tf
from keras.optimizers import SGD
from keras.layers.merge import Concatenate


'''
eleContent:[ [ [batch1 * setl],[batch2 * setl],[batch3 * setl],...,[] ] * 19 types ]
'''

pathTrainSet = "train_set"
pathTestSet = "test_set"

typeList = os.listdir(pathTrainSet)
typeNumber = len(typeList)
graph = 0
constant_retrain = 1
'''
The i(th) sample set to test
'''
with tf.device('/cpu:0'):
  for i in range(1):
  
    # dimensions of our images.
    img_width, img_height = 64, 64

    trainDataDir = pathTrainSet
    testDataDir = pathTestSet
  
    epochs = 25
    batch_size = 180

    nbTrainSamples = 1629
    nbTestSamples = 258

    '''
    move the picture
    '''
  
    '''
    train and test
    '''
    try:
      if constant_retrain:
        axisNo = 3
        input_shape = (img_width, img_height, 3)

        inputs = Input(shape = input_shape)
      
        layer1_1 = Conv2D(32, (4, 4), input_shape=input_shape, strides=(1, 1))(inputs)
        layer1_1 = Activation('relu')(layer1_1)
        layer1_1 = MaxPooling2D(pool_size=(2, 2))(layer1_1)

        final = Flatten()(layer1_1)

        final = Dense(19)(final)
        final = Activation('softmax')(final)

        model = Model(inputs = inputs, outputs = final)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
        graph = tf.get_default_graph()

        # this is the augmentation configuration we will use for training
        trainDatagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        # this is the augmentation configuration we will use for testing:
        # only rescaling
        testDatagen = ImageDataGenerator(rescale=1. / 255)

        trainGenerator = trainDatagen.flow_from_directory(
            trainDataDir,
            target_size=(img_width, img_height),
            batch_size=batch_size)

        testGenerator = testDatagen.flow_from_directory(
            testDataDir,
            target_size=(img_width, img_height),
            batch_size=batch_size)

        model.fit_generator(
            trainGenerator,
            steps_per_epoch=nbTrainSamples // batch_size,
            epochs=epochs,
            validation_data=testGenerator,
            validation_steps=nbTestSamples // batch_size)

        model.save('model')
      else:
        model = load_model('model')
      t = model.evaluate_generator(testGenerator, 1)
      print(t)
      print('bye')
    except Exception as e:
      print("error", e)



print('hello')