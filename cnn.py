import os
import tensorflow as tf

# Importing the required Keras modules containing model and layers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import model_from_json

import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_index = 7777 # You may select anything up to 60,000
# print(y_train[image_index]) # The label is 8
# plt.imshow(x_train[image_index], cmap='Greys')
model = None
input_shape = (28, 28, 1)

def prepare_data_set():
    global x_train, y_train, x_test, y_test
    # Reshaping the array to 4-dims so that it can work with the Keras API
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # Making sure that the values are float so that we can get decimal points after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # Normalizing the RGB codes by dividing it to the max RGB value.
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print('Number of images in x_train', x_train.shape[0])
    print('Number of images in x_test', x_test.shape[0])

def train_cnn() :
    global x_train, y_train, x_test, y_test
    global model
    # Creating a Sequential Model and adding the layers

    if os.path.isfile('cnn_model.json'):
        json_file = open('cnn_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights("cnn_model_weights.h5")
        print("Loaded CNN model from disk")
    else:
        prepare_data_set()

        print("CNN train begin")

        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10,activation=tf.nn.softmax))

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x=x_train,y=y_train, epochs=10)

        model.evaluate(x_test, y_test)

        print("CNN train finish")

        # serialize model to JSON
        model_json = model.to_json()
        with open("cnn_model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("cnn_model_weights.h5")
        print("Saved CNN model to disk")

def predict(img):
    global x_train, y_train, x_test, y_test
    global model
    # image_index = 4444
    # plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
    pred = model.predict(img.reshape(1, 28, 28, 1))
    # print("prepoznato", pred.argmax())
    return pred.argmax()
