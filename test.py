import cv2
from datetime import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os, shutil, itertools, pathlib
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Activation, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, Adamax


def create_model():
    gen_dict = train_gen.class_indices
    classes = list(gen_dict.keys())
    num_class = len(classes)
    image_size = (224, 244)

    image_shape = (image_size[0], image_size[1], 3)

    base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top = False ,
                                                                   weights = 'imagenet' ,
                                                                   input_shape = image_shape,
                                                                   pooling= 'max')
    model = models.Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(256,
              kernel_regularizer=regularizers.l2(l=0.016),
              activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006),
              activation='relu'),
        Dropout(rate=0.4, seed=75),
        Dense(num_class, activation='softmax')
    ])

    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


model = create_model()
model.load_weights('./classification/cp.ckpt')
print('yeah')