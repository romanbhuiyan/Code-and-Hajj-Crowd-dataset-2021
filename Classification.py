########CNN with TFLearn (PIL for Image processing)¶

from PIL import Image
##support for opening, manipulating, and saving many different image file formats

import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
# tf.reset_default_graph()
tf.compat.v1.get_default_graph()
print(tf.compat.v1.get_default_graph())

from tflearn.layers.conv import conv_2d, max_pool_2d
# import pickle

TRAIN_DIR = './DATASET/TRAIN-DATA'
TEST_DIR = './DATASET/TEST-DATA'

IMG_SIZE = 120
LR = 1e-3
#n_epoch = 2
# def label_img(img):
#     word_label = img.split('_')[0]
#     if word_label == 'HIGH':
#         return [1, 0]  # one hot encoding
#     elif word_label == 'LOW':
#         return [0, 1]  # one hot encoding
#     elif word_label == 'VERY_HIGH':
#         return [0, 0]

def label_img(img):
    word_label = img.split('_')[0]
    if word_label == 'HIGH':
        return [0, 0, 0, 1, 0]  # one hot encoding
    elif word_label == 'LOW':
        return [0, 1, 0, 0, 0]  # one hot encoding
    elif word_label == 'VHIGH':
        return [0, 0, 0, 0, 1]  # one hot encoding
    elif word_label == 'MEDIUM':
        return [0, 0, 1, 0, 0]  # one hot encoding
    elif word_label == 'VLOW':
        return [1, 0, 0, 0, 0]  # one hot encoding

def create_train_data():
    train_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        ############################################################
        #    This part is different from sentdex's tutorial
        # Chose to use PIL instead of cv2 for image pre-processing
        ############################################################

        img = Image.open(path)  # Read image syntax with PIL Library
        img = img.convert('L')  # Grayscale conversion with PIL library
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)  # Resizing image syntax with PIL Library

        ############################################################

        train_data.append([np.array(img), np.array(label)])
    shuffle(train_data)
    np.save('train_data.npy', train_data)  # .npy extension = numpy file
    return train_data

train_data = create_train_data()
plt.imshow(train_data[43][0], cmap='gist_gray')
print(train_data[43][1])

def process_test_data():
    test_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        if "DS_Store" not in path:
            img_num = img.split('_')[1]  # images are formatted 'HIGH_2', 'LOW_56'..

            # PIL LIBRARY instead of cv2
            img = Image.open(path)
            img = img.convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

            test_data.append([np.array(img), img_num])
    shuffle(test_data)
    np.save('test_data.npy', test_data)
    return test_data

####### Define CNN (with layers)

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 5, activation='softmax')  # output
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_verbose=3)

train = train_data[-21600:]
test = train_data[:-21600]

##Data preprocessing

X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=35, batch_size=64, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id='HIGH-LOW')

test_data = process_test_data()
fig = plt.figure()

for num, data in enumerate(test_data[:12]):
    img_num = data[1]
    img_data = data[0]

    y = fig.add_subplot(3, 4, num + 1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    print(model_out)
    if np.argmax(model_out) == 1:
        str_label = 'LOW'
    elif np.argmax(model_out) == 2:
        str_label = 'MEDIUM'
    elif np.argmax(model_out) == 3:
        str_label = 'VHIGH'
    elif np.argmax(model_out) == 4:
        str_label = 'HIGH'
    else:
        str_label = 'VLOW'

    y.imshow(orig)
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    plt.savefig("images/density.png")
plt.show()

# , cmap='gray'

