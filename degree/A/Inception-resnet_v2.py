from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers import Dense, Dropout, Lambda, Flatten, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
import datetime
import keras
from keras import optimizers
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

starttime = datetime.datetime.now()
RESNET_V2_A_COUNT = 0
RESNET_V2_B_COUNT = 0
RESNET_V2_C_COUNT = 0


def resnet_v2_stem(x_input):
    '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''

    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    with K.name_scope("stem"):
        x = Conv2D(32, (3, 3), activation="relu", strides=(2, 2))(x_input)  # 149 * 149 * 32
        x = Conv2D(32, (3, 3), activation="relu")(x)  # 147 * 147 * 32
        x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)  # 147 * 147 * 64

        x1 = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x2 = Conv2D(96, (3, 3), activation="relu", strides=(2, 2))(x)

        x = concatenate([x1, x2], axis=-1)  # 73 * 73 * 160

        x1 = Conv2D(64, (1, 1), activation="relu", padding="same")(x)
        x1 = Conv2D(96, (3, 3), activation="relu")(x1)

        x2 = Conv2D(64, (1, 1), activation="relu", padding="same")(x)
        x2 = Conv2D(64, (7, 1), activation="relu", padding="same")(x2)
        x2 = Conv2D(64, (1, 7), activation="relu", padding="same")(x2)
        x2 = Conv2D(96, (3, 3), activation="relu", padding="valid")(x2)

        x = concatenate([x1, x2], axis=-1)  # 71 * 71 * 192

        x1 = Conv2D(192, (3, 3), activation="relu", strides=(2, 2))(x)

        x2 = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = concatenate([x1, x2], axis=-1)  # 35 * 35 * 384

        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x


def inception_resnet_v2_A(x_input, scale_residual=True):
    '''Architecture of Inception_ResNet_A block which is a 35 * 35 grid module.'''
    global RESNET_V2_A_COUNT
    RESNET_V2_A_COUNT += 1
    with K.name_scope('inception_resnet_v2_A' + str(RESNET_V2_A_COUNT)):
        ar1 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)

        ar2 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
        ar2 = Conv2D(32, (3, 3), activation="relu", padding="same")(ar2)

        ar3 = Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
        ar3 = Conv2D(48, (3, 3), activation="relu", padding="same")(ar3)
        ar3 = Conv2D(64, (3, 3), activation="relu", padding="same")(ar3)

        merged = concatenate([ar1, ar2, ar3], axis=-1)

        ar = Conv2D(384, (1, 1), activation="linear", padding="same")(merged)
        if scale_residual: ar = Lambda(lambda a: a * 0.1)(ar)

        x = add([x_input, ar])
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x


def inception_resnet_v2_B(x_input, scale_residual=True):
    '''Architecture of Inception_ResNet_B block which is a 17 * 17 grid module.'''
    global RESNET_V2_B_COUNT
    RESNET_V2_B_COUNT += 1
    with K.name_scope('inception_resnet_v2_B' + str(RESNET_V2_B_COUNT)):
        br1 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)

        br2 = Conv2D(128, (1, 1), activation="relu", padding="same")(x_input)
        br2 = Conv2D(160, (1, 7), activation="relu", padding="same")(br2)
        br2 = Conv2D(192, (7, 1), activation="relu", padding="same")(br2)

        merged = concatenate([br1, br2], axis=-1)

        br = Conv2D(1152, (1, 1), activation="linear", padding="same")(merged)
        if scale_residual: br = Lambda(lambda b: b * 0.1)(br)

        x = add([x_input, br])
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x


def inception_resnet_v2_C(x_input, scale_residual=True):
    '''Architecture of Inception_ResNet_C block which is a 8 * 8 grid module.'''
    global RESNET_V2_C_COUNT
    RESNET_V2_C_COUNT += 1
    with K.name_scope('inception_resnet_v2_C' + str(RESNET_V2_C_COUNT)):
        cr1 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)

        cr2 = Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)
        cr2 = Conv2D(224, (1, 3), activation="relu", padding="same")(cr2)
        cr2 = Conv2D(256, (3, 1), activation="relu", padding="same")(cr2)

        merged = concatenate([cr1, cr2], axis=-1)

        cr = Conv2D(2144, (1, 1), activation="linear", padding="same")(merged)
        if scale_residual: cr = Lambda(lambda c: c * 0.1)(cr)

        x = add([x_input, cr])
        x = BatchNormalization(axis=-1)(x)
        x = Activation("relu")(x)
    return x

def reduction_resnet_v2_A(x_input, k=256, l=256, m=384, n=384):
    with K.name_scope('reduction_resnet_A'):
        ra1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

        ra2 = Conv2D(n, (3, 3), activation='relu', strides=(2, 2), padding='valid')(x_input)

        ra3 = Conv2D(k, (1, 1), activation='relu', padding='same')(x_input)
        ra3 = Conv2D(l, (3, 3), activation='relu', padding='same')(ra3)
        ra3 = Conv2D(m, (3, 3), activation='relu', strides=(2, 2), padding='valid')(ra3)

        merged_vector = concatenate([ra1, ra2, ra3], axis=-1)

        x = BatchNormalization(axis=-1)(merged_vector)
        x = Activation('relu')(x)
    return x

def reduction_resnet_v2_B(x_input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_ResNet_B block.'''
    with K.name_scope('reduction_resnet_v2_B'):
        rbr1 = MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x_input)

        rbr2 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
        rbr2 = Conv2D(384, (3, 3), activation="relu", strides=(2, 2))(rbr2)

        rbr3 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
        rbr3 = Conv2D(288, (3, 3), activation="relu", strides=(2, 2))(rbr3)

        rbr4 = Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
        rbr4 = Conv2D(288, (3, 3), activation="relu", padding="same")(rbr4)
        rbr4 = Conv2D(320, (3, 3), activation="relu", strides=(2, 2))(rbr4)

        merged = concatenate([rbr1, rbr2, rbr3, rbr4], axis=-1)
        rbr = BatchNormalization(axis=-1)(merged)
        rbr = Activation("relu")(rbr)
    return rbr




x_input = Input((299, 299, 3))  # Channels last, as using Tensorflow backend with Tensorflow image dimension ordering

# Input shape is 299 * 299 * 3
x = resnet_v2_stem(x_input)  # Output: 35 * 35 * 256

# 5 x Inception A
for i in range(5):
    x = inception_resnet_v2_A(x, scale_residual=False)
# Output: 35 * 35 * 256

# Reduction A
x = reduction_resnet_v2_A(x, k=256, l=256, m=384, n=384)  # Output: 17 * 17 * 896

# 10 x Inception B
for i in range(10):
    x = inception_resnet_v2_B(x, scale_residual=True)
# Output: 17 * 17 * 896

# Reduction B
x = reduction_resnet_v2_B(x)  # Output: 8 * 8 * 1792

# 5 x Inception C
for i in range(5):
    x = inception_resnet_v2_C(x, scale_residual=True)
# Output: 8 * 8 * 1792

# Average Pooling
x = AveragePooling2D((8, 8))(x)  # Output: 1792

# Dropout
x = Dropout(0.2)(x)  # Keep dropout 0.2 as mentioned in the paper
x = Flatten()(x)  # Output: 1792
x = Dense(13, activation='softmax')(x)

model = Model(inputs=x_input, outputs=x, name='Inception-resnet-v2')
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
train_dir = '/home/tx-lab/dingyue/A/train'
validation_dir = '/home/tx-lab/dingyue/A/validation'
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(299, 299),
        batch_size=36,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(299, 299),
        batch_size=36,
        class_mode='categorical')

MC = keras.callbacks.ModelCheckpoint(filepath='/home/tx-lab/city-planning/A/models/Inception-resnet_v2M.h5',monitor='val_accuracy',
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
      validation_steps=17,
      callbacks=[MC,RL])
model.save('/home/tx-lab/city-planning/A/models/Inception-resnet_v2.h5')
with open('/home/tx-lab/city-planning/A/history/Inception-resnet_v2.txt','w') as f:
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


