from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
import os
import datetime
import keras
from keras import optimizers
import matplotlib.pyplot as plt

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
# import os
# os.environ['CUDA_VISIBLE_DEVICES']='1'
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

starttime = datetime.datetime.now()

model = models.Sequential()
#1
model.add(layers.Conv2D(96, (11, 11),strides=(4, 4),padding='same', activation='relu',
                        input_shape=(227, 227, 3)))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

#2
model.add(layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'))
model.add(layers.BatchNormalization(axis=3))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

#3
model.add(layers.Conv2D(384, (3, 3),strides=(1, 1), padding='same', activation='relu'))
#4
model.add(layers.Conv2D(384, (3, 3),strides=(1, 1), padding='same', activation='relu'))
#5
model.add(layers.Conv2D(256, (3, 3),strides=(1, 1), padding='same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
#FC
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(13, activation='softmax'))
model.summary()

model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer=optimizers.sgd(lr=0.001,momentum=0.9,decay=0.0002,nesterov=True),
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
train_dir = 'D:/A/train'
validation_dir = 'D:/A/validation'
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(227, 227),
        batch_size=24,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(227, 227),
        batch_size=24,
        class_mode='categorical')

MC = keras.callbacks.ModelCheckpoint(filepath='/home/tx-lab/city-planning/A/models/AlexnetM.h5',monitor='val_accuracy',
                                                        verbose=1,
                                                        save_best_only=True,
                                                        save_weights_only=False,
                                                        mode='auto',
                                                        period=1)
RL = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                    factor=0.1,
                                                    patience=4,
                                                    verbose=1,
                                                    mode='auto',
                                                    min_delta=0.000001,
                                                    cooldown=0,
                                                    min_lr=0 )


history = model.fit_generator(
      train_generator,
      steps_per_epoch=None,
      epochs=100,
      validation_data=validation_generator,
      validation_steps=None,
      callbacks=[MC,RL])

model.save('D:\workspace\workspace_python\A\models\Alexnet.h5')
with open('D:\workspace\workspace_python\A\history\Alexnet.txt','w') as f:
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

