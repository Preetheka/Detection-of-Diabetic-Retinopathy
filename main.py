import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import os
base_dir = "DB_Image"
image_size = 224
#Creating DataGenerator
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0,
                                                            shear_range = 0.2,
                                                            zoom_range = 0.2,
                                                            width_shift_range = 0.2,
                                                            height_shift_range = 0.2,
                                                            fill_mode="nearest")
batch_size = 32
train_data = train_datagen.flow_from_directory(os.path.join(base_dir,"train"),
                                               target_size=(image_size,image_size),
                                               batch_size=batch_size,
                                               class_mode="categorical"
                                              )
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale = 1/255.0)
test_data = test_datagen.flow_from_directory(os.path.join(base_dir,"validation"),
                                               target_size=(image_size,image_size),
                                               batch_size=batch_size,
                                               class_mode="categorical"
                                              )

##Part 2
import os
initial_count=0
i = 0
dir = "DB_Image/train/Abnormal"
x="DB_Image/train/Normal"
for path in os.listdir(dir):
    if os.path.isfile(os.path.join(dir, path)):
        initial_count =initial_count + 1
print(initial_count)
for path in os.listdir(x):
    if os.path.isfile(os.path.join(x, path)):
        i = i+1
print(i)



##Part 3

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['Abnormal', 'Normal']
students = [initial_count,i]

plt.bar(langs, students, color ='green',
        width = 0.4)
plt.show()

##Part 4

categories = list(train_data.class_indices.keys())
print(categories)

#Part 5

# VGG Model
# importing required modules
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
import warnings
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

# VGG 16 CNN Architecture
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The Fifth convolution
    tf.keras.layers.Conv2D(512, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),

])
for layer in model.layers:
    layer.trainable = False
model.add(Dense(2, activation='softmax'))
model.summary()

#Part 6
model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(train_data,
                    steps_per_epoch=2,
                    validation_data = test_data,
                    validation_steps=2,
                    epochs=10)

#Part 7
history.history.keys()

#Part 8
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Part 9
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import data
import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from tensorflow.keras.optimizers import Adam
from keras.layers.core import Dense, Flatten
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools


def fast_glcm(img, vmin=0, vmax=255, levels=8, kernel_size=5, distance=1.0, angle=0.0):
    mi, ma = vmin, vmax
    ks = kernel_size
    h, w = img.shape

    # digitize
    bins = np.linspace(mi, ma + 1, levels + 1)
    gl1 = np.digitize(img, bins) - 1

    # make shifted image
    dx = distance * np.cos(np.deg2rad(angle))
    dy = distance * np.sin(np.deg2rad(-angle))
    mat = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float32)
    gl2 = cv2.warpAffine(gl1, mat, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)

    # make glcm
    glcm = np.zeros((levels, levels, h, w), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            mask = ((gl1 == i) & (gl2 == j))
            glcm[i, j, mask] = 1

    kernel = np.ones((ks, ks), dtype=np.uint8)
    for i in range(levels):
        for j in range(levels):
            glcm[i, j] = cv2.filter2D(glcm[i, j], -1, kernel)

    glcm = glcm.astype(np.float32)
    return glcm


def fast_glcm_mean(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm mean
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    mean = np.zeros((h, w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i, j] * i / (levels) ** 2

    return mean


def fast_glcm_std(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm std
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    mean = np.zeros((h, w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            mean += glcm[i, j] * i / (levels) ** 2

    std2 = np.zeros((h, w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            std2 += (glcm[i, j] * i - mean) ** 2

    std = np.sqrt(std2)
    return std


def fast_glcm_contrast(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm contrast
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    cont = np.zeros((h, w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            cont += glcm[i, j] * (i - j) ** 2

    return cont


def fast_glcm_dissimilarity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm dissimilarity
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    diss = np.zeros((h, w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            diss += glcm[i, j] * np.abs(i - j)

    return diss


def fast_glcm_homogeneity(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm homogeneity
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    homo = np.zeros((h, w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            homo += glcm[i, j] / (1. + (i - j) ** 2)

    return homo


def fast_glcm_ASM(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm asm, energy
    '''
    h, w = img.shape
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    asm = np.zeros((h, w), dtype=np.float32)
    for i in range(levels):
        for j in range(levels):
            asm += glcm[i, j] ** 2

    ene = np.sqrt(asm)
    return asm, ene


def fast_glcm_max(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm max
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    max_ = np.max(glcm, axis=(0, 1))
    return max_


def fast_glcm_entropy(img, vmin=0, vmax=255, levels=8, ks=5, distance=1.0, angle=0.0):
    '''
    calc glcm entropy
    '''
    glcm = fast_glcm(img, vmin, vmax, levels, ks, distance, angle)
    pnorm = glcm / np.sum(glcm, axis=(0, 1)) + 1. / ks ** 2
    ent = np.sum(-pnorm * np.log(pnorm), axis=(0, 1))
    return ent


def preprocessing_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


from PIL import Image, ImageFile
from skimage.color import rgb2gray

model = tf.keras.models.load_model('DB_Image/dnn-cnn.h5')
# model = tf.keras.models.load_model('/content/dnn-cnn.h5')
image_path = "DB_Image/train/Abnormal/0a9ec1e99ce4.png"
from keras.preprocessing import image

img1 = image.load_img(image_path)
img1 = np.asarray(img1)
new_img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
img = tf.keras.preprocessing.image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
plt.imshow(new_img)
plt.axis("off")
plt.title("Input Image")
plt.show()
# ---Preprocessing
preprocessed_image = preprocessing_image(image_path)
grayscale = rgb2gray(img1)
plt.imshow(grayscale)

# Gray Scaling the image

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


imgxx = mpimg.imread('DB_Image/train/Abnormal/0a9ec1e99ce4.png')

gray = rgb2gray(imgxx)
plt.title("Grayscale Image")
plt.imshow(gray, cmap=plt.get_cmap('gray'))
plt.axis("off")
# plt.savefig('lena_greyscale.png')
plt.show()

# ---GLCM
glcm_mean = fast_glcm_mean(grayscale)
# ---Classification
img = img / 255.0
prediction = model.predict(img)
print(prediction)
plt.axis("off")
plt.imshow(new_img)
if (prediction.flat[0] == 0.1):
    xx = "No DR"
elif (prediction.flat[0] <= 0.5 and prediction.flat[0] >= 0.2):
    xx = "Mild DR"
else:
    xx = "Severe DR"
plt.title(xx)
# plt.title(categories[np.argmax(prediction)])

