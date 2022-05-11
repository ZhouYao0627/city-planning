#coding=utf-8
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import keras
import os
import datetime
from keras.models import Model
from keras import layers
from keras.layers import Dense,Input,BatchNormalization,Activation,Conv2D,SeparableConv2D,MaxPooling2D
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

starttime = datetime.datetime.now()



inpt = Input(shape=(200,200,3))


#--------------------------#
# Entry flow
#--------------------------#
#--------------------#
# block1
#--------------------#
# 299,299,3 -> 149,149,64
x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(inpt)
x = BatchNormalization(name='block1_conv1_bn')(x)
x = Activation('relu', name='block1_conv1_act')(x)
x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
x = BatchNormalization(name='block1_conv2_bn')(x)
x = Activation('relu', name='block1_conv2_act')(x)

#--------------------#
# block2
#--------------------#
# 149,149,64 -> 75,75,128
residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
x = BatchNormalization(name='block2_sepconv1_bn')(x)
x = Activation('relu', name='block2_sepconv2_act')(x)
x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
x = BatchNormalization(name='block2_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
x = layers.add([x, residual])

#--------------------#
# block3
#--------------------#
# 75,75,128 -> 38,38,256
residual = Conv2D(256, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = Activation('relu', name='block3_sepconv1_act')(x)
x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
x = BatchNormalization(name='block3_sepconv1_bn')(x)
x = Activation('relu', name='block3_sepconv2_act')(x)
x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
x = BatchNormalization(name='block3_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
x = layers.add([x, residual])

#--------------------#
# block4
#--------------------#
# 38,38,256 -> 19,19,728
residual = Conv2D(728, (1, 1), strides=(2, 2),padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = Activation('relu', name='block4_sepconv1_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
x = BatchNormalization(name='block4_sepconv1_bn')(x)
x = Activation('relu', name='block4_sepconv2_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
x = BatchNormalization(name='block4_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
x = layers.add([x, residual])

#--------------------------#
# Middle flow
#--------------------------#
#--------------------#
# block5--block12
#--------------------#
# 19,19,728 -> 19,19,728
for i in range(8):
    residual = x
    prefix = 'block' + str(i + 5)

    x = Activation('relu', name=prefix + '_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
    x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
    x = Activation('relu', name=prefix + '_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
    x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
    x = Activation('relu', name=prefix + '_sepconv3_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
    x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

    x = layers.add([x, residual])

#--------------------------#
# Exit flow
#--------------------------#
#--------------------#
# block13
#--------------------#
# 19,19,728 -> 10,10,1024
residual = Conv2D(1024, (1, 1), strides=(2, 2),
                  padding='same', use_bias=False)(x)
residual = BatchNormalization()(residual)

x = Activation('relu', name='block13_sepconv1_act')(x)
x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
x = BatchNormalization(name='block13_sepconv1_bn')(x)
x = Activation('relu', name='block13_sepconv2_act')(x)
x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
x = BatchNormalization(name='block13_sepconv2_bn')(x)

x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
x = layers.add([x, residual])

#--------------------#
# block14
#--------------------#
# 10,10,1024 -> 10,10,2048
x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
x = BatchNormalization(name='block14_sepconv1_bn')(x)
x = Activation('relu', name='block14_sepconv1_act')(x)

x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
x = BatchNormalization(name='block14_sepconv2_bn')(x)
x = Activation('relu', name='block14_sepconv2_act')(x)

x = GlobalAveragePooling2D(name='avg_pool')(x)
x = Dense(12, activation='softmax', name='predictions')(x)

model = Model(inpt,x,name='inception')
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizers.sgd(lr=0.001,momentum=0.9,decay=0.0002,nesterov=True),
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

test_datagen = ImageDataGenerator(rescale=1./255)
train_dir = '/home/tx-lab/dingyue/B/train'
validation_dir = '/home/tx-lab/dingyue/B/validation'
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(200, 200),
        batch_size=36,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(200, 200),
        batch_size=36,
        class_mode='categorical')

MC = keras.callbacks.ModelCheckpoint(filepath='/home/tx-lab/city-planning/degree/B'
                                              '/models/XceptionM.h5',monitor='val_accuracy',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        mode='auto',
                                                        period=1)
RL = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                    factor=0.1,
                                                    patience=10,
                                                    verbose=1,
                                                    mode='auto',
                                                    min_delta=0.000001,
                                                    cooldown=0,
                                                    min_lr=0 )


history = model.fit_generator(
      train_generator,
      steps_per_epoch=100,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=None,
      callbacks=[MC,RL])
model.save('/home/tx-lab/city-planning/degree/B/models/Xception.h5')
with open('/home/tx-lab/city-planning/degree/B/history/Xception.txt','w') as f:
    f.write(str(history.history))

endtime = datetime.datetime.now()
print( (endtime - starttime).seconds)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
