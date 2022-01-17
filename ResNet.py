from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications.resnet50 import ResNet50
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from sklearn.metrics import multilabel_confusion_matrix, classification_report, confusion_matrix

rn_conv = ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(224, 224, 3))

rn_conv.summary()

image_size = 224
batch_size = 20

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('HajjDataset/Traindata/',
                                          target_size=(image_size,image_size),
                                          batch_size =batch_size,
                                          class_mode='categorical',
                                          shuffle=True)

validation_dataset = test.flow_from_directory('HajjDataset/Testdata/',
                                              target_size=(image_size,image_size),
                                              batch_size =batch_size,
                                              class_mode='categorical',
                                              shuffle=False)

print(train_dataset.class_indices)
# print(test_dataset.class_indices)

print(validation_dataset.class_indices)
print(validation_dataset.classes)

nTrain = 21600
nVal = 5400

train_features = np.zeros(shape=(nTrain, 7, 7, 2048))
train_labels = np.zeros(shape=(nTrain,5))



# save the features extracted from the training dataset to be passed to model.fit for training
train_features = np.reshape(train_features, (nTrain, 7 * 7 * 2048))

validation_features = np.zeros(shape=(nVal, 7, 7, 2048))
validation_labels = np.zeros(shape=(nVal,5))

i = 0
for inputs_batch, labels_batch in validation_dataset:
    features_batch = rn_conv.predict(inputs_batch)
    validation_features[i * batch_size : (i + 1) * batch_size] = features_batch
    validation_labels[i * batch_size : (i + 1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= nVal:
        break

# save the features extracted from the validation dataset to be passed to model.fit for training
validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 2048))

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 2048))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-4),
              loss='categorical_crossentropy',
              metrics=['acc'])

history = model.fit(train_features,
                    train_labels,
                    epochs=100,
                    validation_data=(validation_features,validation_labels))

# Plot the accuracy and loss curves
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#----------AddedRoman---------
scores = model.evaluate(train_features, train_labels, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
preds=np.round(model.predict(validation_features), 0)
print('round test labels', preds)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#print('\nAccuracy: {:.2f}\n'.format(accuracy_score(validation_labels, preds)))
print('Precision: {:.2f}'.format(precision_score(validation_labels, preds, average='micro')))
print('Recall: {:.2f}'.format(recall_score(validation_labels, preds, average='micro')))
print('F1-score: {:.2f}\n'.format(f1_score(validation_labels, preds, average='micro')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(validation_labels, preds, target_names=['VLOW', 'LOW', 'MEDIUM', 'HIGH', 'VHIGH']))


#change dir based on category
test_dataset = test.flow_from_directory('HajjDataset/basedata_test_verylow', target_size=(image_size,image_size), batch_size =batch_size, class_mode='categorical', shuffle=False)

nVal = 80

test_features = np.zeros(shape=(nVal, 7, 7, 2048))
test_labels = np.zeros(shape=(nVal,5))

#if high density, test label=[1,0,0,0,0]
#if low density, test label=[0,1,0,0,0]
#if medium density, test label=[0,0,1,0,0]
#if very high density, test label=[0,0,0,1,0]
#if very low density, test label=[0,0,0,0,1]
i = 0
for inputs_batch, labels_batch in test_dataset:
    features_batch = rn_conv.predict(inputs_batch)
    test_features[i * batch_size : (i + 1) * batch_size] = features_batch
    test_labels[i * batch_size : (i + 1) * batch_size] = [0,0,0,0,1]
    i += 1
    if i * batch_size >= nVal:
        break

# save the features extracted from the validation dataset to be passed to model.fit for training
test_features = np.reshape(test_features, (nVal, 7 * 7 * 2048))

score = model.evaluate(test_features, test_labels)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

# if high density, ground_truth=0
# if low density, ground_truthl=1
# if medium density, ground_truth=2
# if very high density, ground_truth=3
# if very low density, ground_truth=4
# Get the filenames from the generator
fnames = test_dataset.filenames

# Get the ground truth from generator
ground_truth = 4

# Get the label to class mapping from the generator
label2index = validation_dataset.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v, k) for k, v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict(test_features)
prob = model.predict(test_features)
# Show the errors
errors = np.where(predictions != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),nVal))
#print("Accuracy = {}/{}".format((nVal-errors)/nVal)
   

    # save the features extracted from the validation dataset to be passed to model.fit for training
    validation_features = np.reshape(validation_features, (nVal2, 7 * 7 * 2048))

    score = model.evaluate(validation_features, validation_labels, verbose=0)
    print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

    model.save("ResNet.h5")

