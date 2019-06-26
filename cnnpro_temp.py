#applying CNN

#importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#initializing CNN
model = Sequential()

#adding layer 1st Convolutional layer
model.add(Conv2D(64,(3,3),
                 activation = 'relu',
                 input_shape = (64,64,1)))
#applying max pooling
model.add(MaxPooling2D(pool_size = (2,2)))


#adding flaten layer
model.add(Flatten())

#applying full connection
#creating artificail nueral network of 2 hidden layer
model.add(Dense(units = 128,
                activation = 'relu'))
model.add(Dense(units = 1,
                activation = 'sigmoid'))

#complinig results
model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])

#fitting cnn
#Code snippet from www.keras.io
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory('train',
                                          target_size=(64, 64),
                                          batch_size=32,
                                          class_mode='binary',
                                          color_mode = 'grayscale')

test = test_datagen.flow_from_directory('test',
                                        target_size=(64, 64),
                                        batch_size=32,
                                        class_mode='binary',
                                        color_mode = 'grayscale')

#saving epoch history
history = model.fit_generator(train,
                    steps_per_epoch=5216,
                    epochs=5,
                    validation_data=test,
                    validation_steps=624)

#plottnig grpah

from keras.utils import plot_model
plot_model(model, to_file = 'model.png')

#saving wieghts 
model.save_weights('model_wieght.h5')

#visualizing results
import matplotlib.pyplot as plt

# Plot train & test accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot train & test loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

