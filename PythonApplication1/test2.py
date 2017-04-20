import os
import io
import math
import codecs
import numpy as np
import string
import shutil
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
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
eleLength = {}
eleContent = []
fileSum = 0
graph = 0

for x in range(typeNumber):
  kind = typeList[x]
  eleContent.append(os.listdir(pathTrainSet+"/"+kind))
  eleLength[kind] = len(eleContent[x])
  fileSum += eleLength[kind]

for eleno in range(len(eleContent)):
  a = eleContent[eleno]
  l = len(a)
  setl = int(l/10)
  np.random.shuffle(a)
  rest = [a[i:] for i in range(setl*10,l,10)]
  eleContent[eleno] = [a[i:i+setl] for i in range(0,l,setl)]
  for el in range(len(rest)):
    eleContent[eleno][el].extend(rest[el])
'''
The i(th) sample set to test
'''
with tf.device('/cpu:0'):
  for i in range(1):
  
    # dimensions of our images.
    img_width, img_height = 150, 150

    trainDataDir = pathTrainSet
    testDataDir = pathTestSet
  
    epochs = 5
    batch_size = 90

    nbTrainSamples = fileSum
    nbTestSamples = 0

    '''
    move the picture
    '''
    for x in range(typeNumber):
      kind = typeList[x]
      pathFrom = pathTrainSet + "/" + kind + "/"
      pathTo = pathTestSet + "/" + kind + "/"
      moveFile = eleContent[x][i]
      for y in moveFile:
        shutil.move(pathFrom+y, pathTo+y)
      nbTrainSamples -= len(moveFile)
      nbTestSamples += len(moveFile)
  
    '''
    train and test
    '''
    try:
      axisNo = 1
      if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
        axisNo = 1
      else:
        input_shape = (img_width, img_height, 3)
        axisNo = 3

      inputs = Input(shape = input_shape)
      
      layer1_1 = Conv2D(48, (11, 11), input_shape=input_shape, strides=(4, 4))(inputs)
      layer1_1 = Activation('relu')(layer1_1)
      #layer1_1 = MaxPooling2D(pool_size=(2, 2))(layer1_1)

      layer1_2 = Conv2D(48, (11, 11), input_shape=input_shape, strides=(4, 4))(inputs)
      layer1_2 = Activation('relu')(layer1_2)
      #layer1_2 = MaxPooling2D(pool_size=(2, 2))(layer1_2)

      layer2_1 = Conv2D(128, (5, 5))(layer1_1)
      layer2_1 = Activation('relu')(layer2_1)
      #layer2_1 =  Conv2D(96, (3, 3))(layer2_1)
      #layer2_1 = Activation('relu')(layer2_1)
      # layer2_1 = MaxPooling2D(pool_size=(2, 2))(layer2_1)

      layer2_2 = Conv2D(128, (5, 5))(layer1_2)
      layer2_2 = Activation('relu')(layer2_2)
      #layer2_2 =  Conv2D(96, (3, 3))(layer2_2)
      #layer2_2 = Activation('relu')(layer2_2)
      # layer2_2 = MaxPooling2D(pool_size=(2, 2))(layer2_2)

      layer3_1 = Concatenate(axis = axisNo)([layer2_1, layer2_2])
      layer3_2 = Concatenate(axis = axisNo)([layer2_2, layer2_1])

      #layer4_1 = Conv2D(192, (3, 3))(layer3_1)
      #layer4_1 = Activation('relu')(layer4_1)

      #layer4_2 = Conv2D(192, (3, 3))(layer3_2)
      #layer4_2 = Activation('relu')(layer4_2)

      layer5_1 = Conv2D(128, (3, 3))(layer4_1)
      layer5_1 = Activation('relu')(layer5_1)

      layer5_2 = Conv2D(128, (3, 3))(layer4_2)
      layer5_2 = Activation('relu')(layer5_2)

      layer6_1 = Flatten()(layer5_1)
      layer6_2 = Flatten()(layer5_2)

      layer6_1_ = Concatenate()(layer6_1, layer6_2)
      layer6_1_ = Dense(256)(layer6_1_)
      layer6_1_ = Activation('relu')(layer6_1_)
      
      layer6_2_ = Concatenate()(layer6_2, layer6_1)
      layer6_2_ = Dense(256)(layer6_2_)
      layer6_2_ = Activation('relu')(layer6_2_)

      layer7_1 = Concatenate()(layer6_1_, layer6_2_)
      layer7_1 = Dense(128)(layer7_1)
      layer7_1 = Activation('relu')(layer7_1)
      
      layer7_2 = Concatenate()(layer6_2_, layer6_1_)
      layer7_2 = Dense(64)(layer7_2)
      layer7_2 = Activation('relu')(layer7_2)

      final = Concatenate()(layer7_1, layer7_2)
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

      model.save_weights('first_try' + str(i) + '.h5')
      t = model.predict_generator(testGenerator,2)
      compResult = [np.argmax(t[i]) for i in range(len(t))]
      compRate = np.mean(compResult == testGenerator.classes)
    except Exception as e:
      print("error", e)
    finally:
      for x in range(typeNumber):
        kind = typeList[x]
        pathFrom = pathTestSet + "/" + kind
        pathTo = pathTrainSet + "/" + kind
        moveFile = os.listdir(pathFrom)
        if len(moveFile) == 0:
          continue
        for y in moveFile:
          shutil.move(pathFrom+"/"+y, pathTo+"/"+y)
  
    pass

print('hello')