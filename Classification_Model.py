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

def get_label_df(path):
    train_path = path
    file_path, labels = list(), list()

    folders = os.listdir(train_path);
    for fold in folders:
        folder_path = os.path.join(train_path, fold)
        files = os.listdir(folder_path)
        for file in files:
            file_path.append(os.path.join(folder_path, file))
            labels.append(fold)

    folder_series = pd.Series(file_path, name="file_paths")
    label_series = pd.Series(labels, name="labels")
    df = pd.concat([folder_series, label_series], axis=1)
    return df

def split_data():
    train_path = '../data/classifyData/Training'
    test_path = '../data/classifyData/Testing'
    train_df, test_df = get_label_df(train_path), get_label_df(test_path)
    train, valid = train_test_split(train_df,
                                train_size = 0.8,
                                shuffle = True,
                                random_state = 42
                               )
    image_size = (224, 244)
    batch_size = 16
    tr_gen, ts_gen = ImageDataGenerator(), ImageDataGenerator()

    train_gen = tr_gen.flow_from_dataframe(train ,
                                           x_col = 'file_paths' ,
                                           y_col = 'labels' ,
                                           target_size = image_size ,
                                          class_mode = 'categorical' ,
                                           color_mode = 'rgb' ,
                                           shuffle = True ,
                                           batch_size =batch_size)

    valid_gen = tr_gen.flow_from_dataframe(valid ,
                                           x_col = 'file_paths' ,
                                           y_col = 'labels' ,
                                           target_size = image_size ,
                                           class_mode = 'categorical',
                                           color_mode = 'rgb' ,
                                           shuffle= True,
                                           batch_size = batch_size)

    test_gen = ts_gen.flow_from_dataframe(test_df ,
                                          x_col= 'file_paths' ,
                                          y_col = 'labels' ,
                                          target_size = image_size ,
                                          class_mode = 'categorical' ,
                                          color_mode= 'rgb' ,
                                          shuffle = False ,
                                          batch_size = batch_size)
    return train_gen, valid_gen, test_gen, image_size


def model_train(train_gen, valid_gen, test_gen, image_size, epoch_num=10):
    checkpoint_path = "./classification/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    gen_dict = train_gen.class_indices
    classes = list(gen_dict.keys())
    image_shape = (image_size[0] , image_size[1] , 3)
    num_class = len(classes)
    logdir = "./logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top = False ,
                                                                   weights = 'imagenet' ,
                                                                   input_shape = image_shape,
                                                                   pooling= 'max')
    model = models.Sequential([
        base_model,
        BatchNormalization(axis= -1 , momentum= 0.99 , epsilon= 0.001),
        Dense(256,
              kernel_regularizer = regularizers.l2(l= 0.016) ,
              activity_regularizer = regularizers.l1(0.006),
                bias_regularizer= regularizers.l1(0.006) ,
              activation = 'relu'),
        Dropout(rate= 0.4 , seed = 75),
        Dense(num_class , activation = 'softmax')
    ])

    model.compile(Adamax(learning_rate = 0.001) , loss = 'categorical_crossentropy', metrics = ['accuracy'])

    history = model.fit(x= train_gen ,
                        epochs = epoch_num ,
                        verbose = 1 ,
                        validation_data = valid_gen ,
                        validation_steps = None ,
                        shuffle = False,
                        callbacks=[tensorboard_callback, cp_callback]
                       )

    return history



def plot_result(history):
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']

    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]

    index_acc = np.argmax(val_acc)
    val_highest = val_acc[index_acc]

    Epochs = [i+1 for i in range(len(train_acc))]

    loss_label = f'Best epochs = {str(index_loss +1)}'
    acc_label = f'Best epochs = {str(index_acc + 1)}'

    #Training history

    plt.figure(figsize= (20,8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1,2,1)
    plt.plot(Epochs , train_loss , 'r' , label = 'Training Loss')
    plt.plot(Epochs , val_loss , 'g' , label = 'Validation Loss')
    plt.scatter(index_loss + 1 , val_lowest , s = 150 , c = 'blue',label = loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./training_validation_loss.png', dpi=400)

    plt.subplot(1,2,2)
    plt.plot(Epochs , train_acc , 'r' , label = 'Training Accuracy')
    plt.plot(Epochs , val_acc , 'g' , label = 'Validation Accuracy')
    plt.scatter(index_acc + 1 , val_highest , s = 150 , c = 'blue',label = acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show();
    plt.savefig('./training_validation_accuracy.png', dpi=400)


def test_model(test_gen, model):

    preds = model.predict_generator(test_gen)

    y_pred = np.argmax(preds , axis = 1)

    g_dict = test_gen.class_indices
    classes = list(g_dict.keys())

    # Confusion matrix
    cm = confusion_matrix(test_gen.classes, y_pred)

    plt.figure(figsize= (10, 10))
    plt.imshow(cm, interpolation= 'nearest', cmap= plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation= 45)
    plt.yticks(tick_marks, classes)


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment= 'center', color= 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.show()
    plt.savefig('prediction.png', dpi=400)

    #Classification Report
    print(classification_report(test_gen.classes, y_pred , target_names= classes ))

def create_model(train_gen):
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
    return model




if __name__ == "__main__":
    train_gen, valid_gen, test_gen, image_size = split_data()
    history = model_train(train_gen, valid_gen, test_gen, image_size, epoch_num=20)
    plot_result(history)



