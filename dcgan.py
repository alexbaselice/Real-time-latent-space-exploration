from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy as np
from PIL import ImageTk, Image
import argparse
import math

import tkinter as tk
from tkinter import Label, Button
import time



class Application:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Real Time Latent Space Viewer')
        self.latent_vector = np.zeros(shape=(1, 100))

        self.panel = Label(self.root, text="Loading Image")
        self.panel.pack(side="bottom", fill="both", expand="yes")

        self.slider = tk.Scale(self.root, from_=-DIM_AMOUNT, to=DIM_AMOUNT,
                               orient="horizontal")
        self.slider.bind("<ButtonRelease-1>", self.updateValue)
        self.slider.pack()

        self.slider2 = tk.Scale(self.root, from_=-DIM_AMOUNT, to=DIM_AMOUNT,
                               orient="horizontal")
        self.slider2.bind("<ButtonRelease-1>", self.updateValue2)
        self.slider2.pack()

        self.inter_button = Button(self.root, text="Iterate")
        self.inter_button.bind("<ButtonRelease-1>", self.iterate_dim)
        self.start = -100
        self.inter_button.pack()


    def iterate_dim(self, event):
        for index in range(-100,100):
            num = index

            print (self.latent_vector)
            for i in range(self.latent_vector.shape[-1]):
                if i == 0:
                    self.latent_vector[:, i] = num
            print (self.latent_vector)

            img = generateImage(1, latent_vector=self.latent_vector)
            self.panel.configure(image=img)
            self.panel.image = img
            self.root.update_idletasks()

    def updateValue(self, event):
        num = self.slider.get()

        print (self.latent_vector)
        for i in range(self.latent_vector.shape[-1]):
            if i == 0:
                self.latent_vector[:, i] = num
        print (self.latent_vector)

        img = generateImage(1, latent_vector=self.latent_vector)
        self.panel.configure(image=img)
        self.panel.image = img

    def updateValue2(self, event):
        num = self.slider2.get()

        print (self.latent_vector)
        for i in range(self.latent_vector.shape[-1]):
            if i == 1:
                self.latent_vector[:, i] = num
        print (self.latent_vector)

        img = generateImage(1, latent_vector=self.latent_vector)
        self.panel.configure(image=img)
        self.panel.image = img



def generator_model():
    model = Sequential()
    model.add(Dense(input_dim=100, output_dim=1024))
    model.add(Activation('tanh'))
    model.add(Dense(128*7*7))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(64, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(1, (5, 5), padding='same'))
    model.add(Activation('tanh'))
    return model


def discriminator_model():
    model = Sequential()
    model.add(
            Conv2D(64, (5, 5),
            padding='same',
            input_shape=(28, 28, 1))
            )
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (5, 5)))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model


def generator_containing_discriminator(g, d):
    model = Sequential()
    model.add(g)
    d.trainable = False
    model.add(d)
    return model


def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:3]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[:, :, 0]
    return image


def train(BATCH_SIZE):
    #Old arrays
    (X_train_all, y_train_all), (X_test_all, y_test_all) = mnist.load_data()
    # Single num array
    X_train, y_train, X_test, y_test = [],[],[],[]
    # Deleting all but one digit
    for index in range(60000):
        if y_train_all[index] == ONLY_DIGIT_WANTED:
            X_train.append(X_train_all[index])
            y_train.append(y_train_all[index])
    for index in range(10000):
        if y_test_all[index] == ONLY_DIGIT_WANTED:
            X_test.append(X_test_all[index])
            y_test.append(y_test_all[index])


    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print(X_train)
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train[:, :, :, None]
    X_test = X_test[:, :, :, None]
    # X_train = X_train.reshape((X_train.shape, 1) + X_train.shape[1:])
    d = discriminator_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    d_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g_optim = SGD(lr=0.0005, momentum=0.9, nesterov=True)
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    d_on_g.compile(loss='binary_crossentropy', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='binary_crossentropy', optimizer=d_optim)
    for epoch in range(EPOCHS):
        print("Epoch is", epoch)
        print("Number of batches", int(X_train.shape[0]/BATCH_SIZE))
        for index in range(int(X_train.shape[0]/BATCH_SIZE)):
            noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            if index % 50 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                Image.fromarray(image.astype(np.uint8)).save(
                    "images/"+str(epoch)+"_"+str(index)+".png")
            X = np.concatenate((image_batch, generated_images))
            y = [1] * BATCH_SIZE + [0] * BATCH_SIZE
            d_loss = d.train_on_batch(X, y)
            print("batch %d d_loss : %f" % (index, d_loss))
            noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
            d.trainable = False
            g_loss = d_on_g.train_on_batch(noise, [1] * BATCH_SIZE)
            d.trainable = True
            print("batch %d g_loss : %f" % (index, g_loss))
            if index % 10 == 9:
                g.save_weights('generator', True)
                d.save_weights('discriminator', True)


def generate(BATCH_SIZE, nice=False):
    g = generator_model()
    g.compile(loss='binary_crossentropy', optimizer="SGD")
    g.load_weights('generator')
    if nice:
        d = discriminator_model()
        d.compile(loss='binary_crossentropy', optimizer="SGD")
        d.load_weights('discriminator')
        noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))
        generated_images = g.predict(noise, verbose=1)
        d_pret = d.predict(generated_images, verbose=1)
        index = np.arange(0, BATCH_SIZE*20)
        index.resize((BATCH_SIZE*20, 1))
        pre_with_index = list(np.append(d_pret, index, axis=1))
        pre_with_index.sort(key=lambda x: x[0], reverse=True)
        nice_images = np.zeros((BATCH_SIZE,) + generated_images.shape[1:3], dtype=np.float32)
        nice_images = nice_images[:, :, :, None]
        for i in range(BATCH_SIZE):
            idx = int(pre_with_index[i][1])
            nice_images[i, :, :, 0] = generated_images[idx, :, :, 0]
        image = combine_images(nice_images)
    else:
        noise = np.random.uniform(-1, 1, (BATCH_SIZE, 100))
        generated_images = g.predict(noise, verbose=1)
        image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
        "generated_image.png")

def realtime(BATCH_SIZE, nice=False):

    app = Application()
    image = generateImage(1, latent_vector=app.latent_vector)
    app.panel.configure(image=image)
    app.panel.image = image
    app.root.mainloop()


def generateImage(BATCH_SIZE, latent_vector):
    # noise = np.random.uniform(-1, 1, (BATCH_SIZE*20, 100))

    generated_image = g.predict(latent_vector, verbose=1)
    image = combine_images(generated_image)
    image = image * 127.5 + 127.5
    image = Image.fromarray(image.astype(np.uint8))
    image = ImageTk.PhotoImage(image)
    return image

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--nice", dest="nice", action="store_true")
    parser.set_defaults(nice=False)
    args = parser.parse_args()
    return args

EPOCHS = 10
ONLY_DIGIT_WANTED = 1
DIM_AMOUNT = 100

g = generator_model()
g.compile(loss='binary_crossentropy', optimizer="SGD")
g.load_weights('generator')

d = discriminator_model()
d.compile(loss='binary_crossentropy', optimizer="SGD")
d.load_weights('discriminator')


if __name__ == "__main__":
    args = get_args()
    if args.mode == "train":
        train(BATCH_SIZE=args.batch_size)
    elif args.mode == "generate":
        generate(BATCH_SIZE=args.batch_size, nice=args.nice)
    elif args.mode == "realtime":
        realtime(BATCH_SIZE=args.batch_size, nice=args.nice)
