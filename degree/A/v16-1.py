#只更改卷积方式
from keras.layers import add,Flatten,MaxPooling2D,Dropout
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import keras
import datetime
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.layers import Input
from keras.layers import ZeroPadding2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import DepthwiseConv2D
from keras.layers import Activation,Dense,AveragePooling2D,GlobalAvgPool2D,MaxPooling2D
from keras.models import Model
import numpy as np

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

starttime = datetime.datetime.now()

def zeropadding2d(inputs, input_shape=None, padding = ((1,0),(0,1))):
    if input_shape==None:
        x = ZeroPadding2D(padding=padding)(inputs)
    else:
        x = ZeroPadding2D(input_shape = input_shape, padding=padding)(inputs)
    return x

def conv2d(inputs=None, filters=32, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False,Activation='relu'):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)(inputs)
    x = BatchNormalization()(x)
    return x

def depthwiseconv2d(inputs, strides=(1,1), kernel_size=(3,3), use_bias=False,Activation='relu'):
    if strides[0] == 1:
        padding = 'same'
    else:
        padding = 'valid'
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding, use_bias = use_bias)(inputs)
    x = BatchNormalization()(x)
    return x
seed = 7
np.random.seed(seed)
INPUT = Input(shape=(224,224,3))
#conv2d_1
x = conv2d(inputs=INPUT, filters=64, kernel_size=(3,3))
#conv2d_2
x = depthwiseconv2d(x,strides=(1,1))
x = conv2d(x,filters=64)
#zeropadding_1
x = MaxPooling2D(pool_size=(2,2))(x)
#maxpooling_1
x = depthwiseconv2d(x,strides=(1,1))
#conv2d_3
x = conv2d(x,filters=128)
#conv2d_4
x = depthwiseconv2d(x,strides=(1,1))
x = conv2d(x,filters=128)
#zeropadding_2
x = MaxPooling2D(pool_size=(2,2))(x)
#maxpooling_2
x = depthwiseconv2d(x,strides=(1,1))
#conv2d_5
x = conv2d(x,filters=256)
#conv2d_6
x = depthwiseconv2d(x,strides=(1,1))
x = conv2d(x,filters=256)
#conv2d_7
x = depthwiseconv2d(x,strides=(1,1))
x = conv2d(x,filters=256)
#zeropadding_3
x = MaxPooling2D(pool_size=(2,2))(x)
#maxpooling_3
x = depthwiseconv2d(x,strides=(1,1))
#conv2d_8
x = conv2d(x,filters=512)
#conv2d_9
x = depthwiseconv2d(x,strides=(1,1))
x = conv2d(x,filters=512)
#conv2d_10
x = depthwiseconv2d(x,strides=(1,1))
x = conv2d(x,filters=512)
#zeropadding_4
x = MaxPooling2D(pool_size=(2,2))(x)
#maxpooling_4
x = depthwiseconv2d(x,strides=(1,1))
#conv2d_11
x = conv2d(x,filters=512)
#conv2d_12
x = depthwiseconv2d(x,strides=(1,1))
x = conv2d(x,filters=512)
#conv2d_13
x = depthwiseconv2d(x,strides=(1,1))
x = conv2d(x,filters=512)
#maxpooling_5
x = MaxPooling2D(pool_size=(2,2))(x)
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(13,activation='softmax')(x)
model = Model(INPUT, x)
model.summary()

# model.compile(loss=keras.losses.categorical_crossentropy,
#               optimizer=optimizers.sgd(lr=0.001,momentum=0.9,decay=0.0002,nesterov=True),
#               metrics=['accuracy'])
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,)
#
# test_datagen = ImageDataGenerator(rescale=1./255)
# train_dir = '/home/tx-lab/dingyue/A/train'
# validation_dir = '/home/tx-lab/dingyue/A/validation'
# train_generator = train_datagen.flow_from_directory(
#         train_dir,
#         target_size=(224, 224),
#         batch_size=24,
#         class_mode='categorical')
#
# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(224, 224),
#         batch_size=24,
#         class_mode='categorical')
#
# MC = keras.callbacks.ModelCheckpoint(filepath='/home/tx-lab/city-planning/A/models/v16-1M.h5',monitor='val_accuracy',
#                                                         verbose=1,
#                                                         save_best_only=True,
#                                                         save_weights_only=False,
#                                                         mode='auto',
#                                                         period=1)
# RL = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
#                                                     factor=0.1,
#                                                     patience=10,
#                                                     verbose=1,
#                                                     mode='auto',
#                                                     min_delta=0.000001,
#                                                     cooldown=0,
#                                                     min_lr=0 )
#
#
# history = model.fit_generator(
#       train_generator,
#       steps_per_epoch=None,
#       epochs=100,
#       validation_data=validation_generator,
#       validation_steps=None,
#       callbacks=[MC,RL])
#
# model.save('/home/tx-lab/city-planning/A/models/v16-1.h5')
# with open('/home/tx-lab/city-planning/A/history/v16-1.txt','w') as f:
#     f.write(str(history.history))
#
# endtime = datetime.datetime.now()
# print( (endtime - starttime).seconds)
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()






































