from keras.layers import Input
from keras.layers.merge import concatenate, add
from keras.layers import Dense, Dropout, Lambda, Flatten, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import backend as K
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
RESNET_V1_A_COUNT = 0
RESNET_V1_B_COUNT = 0
RESNET_V1_C_COUNT = 0


def resnet_v1_stem(x_input):
    with K.name_scope('Stem'):
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid')(x_input)
        x = Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)
        x = Conv2D(80, (1, 1), activation='relu', padding='same')(x)
        x = Conv2D(192, (3, 3), activation='relu', padding='valid')(x)
        x = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(x)
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
    return x


def inception_resnet_v1_A(x_input, scale_residual=True):
    """ 35x35 卷积核"""
    global RESNET_V1_A_COUNT
    RESNET_V1_A_COUNT += 1
    with K.name_scope('resnet_v1_A' + str(RESNET_V1_A_COUNT)):
        ar1 = Conv2D(32, (1, 1), activation='relu', padding='same')(x_input)

        ar2 = Conv2D(32, (1, 1), activation='relu', padding='same')(x_input)
        ar2 = Conv2D(32, (3, 3), activation='relu', padding='same')(ar2)

        ar3 = Conv2D(32, (1, 1), activation='relu', padding='same')(x_input)
        ar3 = Conv2D(32, (3, 3), activation='relu', padding='same')(ar3)
        ar3 = Conv2D(32, (3, 3), activation='relu', padding='same')(ar3)

        merged_vector = concatenate([ar1, ar2, ar3], axis=-1)

        ar = Conv2D(256, (1, 1), activation='linear', padding='same')(merged_vector)

        if scale_residual:  # 是否缩小
            ar = Lambda(lambda x: 0.1*x)(ar)
        x = add([x_input, ar])
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
    return x


def inception_resnet_v1_B(x_input, scale_residual=True):
    """ 17x17 卷积核"""
    global RESNET_V1_B_COUNT
    RESNET_V1_B_COUNT += 1
    with K.name_scope('resnet_v1_B' + str(RESNET_V1_B_COUNT)):
        br1 = Conv2D(128, (1, 1), activation='relu', padding='same')(x_input)

        br2 = Conv2D(128, (1, 1), activation='relu', padding='same')(x_input)
        br2 = Conv2D(128, (1, 7), activation='relu', padding='same')(br2)
        br2 = Conv2D(128, (7, 1), activation='relu', padding='same')(br2)

        merged_vector = concatenate([br1, br2], axis=-1)

        br = Conv2D(896, (1, 1), activation='linear', padding='same')(merged_vector)

        if scale_residual:
            br = Lambda(lambda x: 0.1*x)(br)
        x = add([x_input, br])
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)

    return x


def inception_resnet_v1_C(x_input, scale_residual=True):
    global RESNET_V1_C_COUNT
    RESNET_V1_C_COUNT += 1
    with K.name_scope('resnet_v1_C' + str(RESNET_V1_C_COUNT)):
        cr1 = Conv2D(192, (1, 1), activation='relu', padding='same')(x_input)

        cr2 = Conv2D(192, (1, 1), activation='relu', padding='same')(x_input)
        cr2 = Conv2D(192, (1, 3), activation='relu', padding='same')(cr2)
        cr2 = Conv2D(192, (3, 1), activation='relu', padding='same')(cr2)

        merged_vector = concatenate([cr1, cr2], axis=-1)

        cr = Conv2D(1792, (1, 1), activation='relu', padding='same')(merged_vector)

        if scale_residual:
            cr = Lambda(lambda x: 0.1*x)(cr)
        x = add([x_input, cr])
        x = BatchNormalization(axis=-1)(x)
        x = Activation('relu')(x)
    return x


def reduction_resnet_A(x_input, k=192, l=224, m=256, n=384):
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


def reduction_resnet_B(x_input):
    with K.name_scope('reduction_resnet_B'):
        rb1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),padding='valid')(x_input)

        rb2 = Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
        rb2 = Conv2D(384, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb2)

        rb3 = Conv2D(256, (1, 1),activation='relu', padding='same')(x_input)
        rb3 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb3)

        rb4 = Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
        rb4 = Conv2D(256, (3, 3), activation='relu', padding='same')(rb4)
        rb4 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb4)

        merged_vector = concatenate([rb1, rb2, rb3, rb4], axis=-1)

        x = BatchNormalization(axis=-1)(merged_vector)
        x = Activation('relu')(x)
    return x



x_input = Input(shape=(299, 299, 3))
# stem
x = resnet_v1_stem(x_input)

# 5 x inception_resnet_v1_A
for i in range(5):
    x = inception_resnet_v1_A(x, scale_residual=False)

# reduction_resnet_A
x = reduction_resnet_A(x, k=192, l=192, m=256, n=384)

# 10 x inception_resnet_v1_B
for i in range(10):
    x = inception_resnet_v1_B(x, scale_residual=True)

# Reduction B
x = reduction_resnet_B(x)

# 5 x Inception C
for i in range(5):
    x = inception_resnet_v1_C(x, scale_residual=True)

# Average Pooling
x = AveragePooling2D(pool_size=(8, 8))(x)

# dropout
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(13, activation='softmax')(x)

model = Model(inputs=x_input, outputs=x, name='Inception-resnet-v1')
model.summary()
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizers.sgd(lr=0.0008,momentum=0.9,decay=0.0002,nesterov=True),
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

MC = keras.callbacks.ModelCheckpoint(filepath='/home/tx-lab/city-planning/A/models/Inception-resnet_v1M.h5',monitor='val_accuracy',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        mode='auto',
                                                        period=1)
RL = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                    factor=0.1,
                                                    patience=6,
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
model.save('/home/tx-lab/city-planning/A/models/Inception-resnet_v1.h5')
with open('/home/tx-lab/city-planning/A/history/Inception-resnet_v1.txt','w') as f:
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


