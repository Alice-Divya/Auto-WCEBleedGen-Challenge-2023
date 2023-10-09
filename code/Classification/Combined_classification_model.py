#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, ResNet101
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

# root directory path
root_folder = 'Dataset path'

# loading data
def load_and_preprocess_data(data_folder, preprocessing_function):
    datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    generator = datagen.flow_from_directory(
        data_folder,
        target_size=(224, 224),  
        batch_size=32,
        class_mode='binary'
    )
    return generator

# training data
train_generator = load_and_preprocess_data(os.path.join(root_folder, 'train'), tf.keras.applications.vgg16.preprocess_input)

# validation data
val_generator = load_and_preprocess_data(os.path.join(root_folder, 'validation'), tf.keras.applications.vgg16.preprocess_input)

# VGG16 
def create_vgg_model():
    base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x_vgg = base_model_vgg.output
    x_vgg = GlobalAveragePooling2D()(x_vgg)
    x_vgg = Dense(1024, activation='relu')(x_vgg)
    predictions_vgg = Dense(1, activation='sigmoid')(x_vgg)
    model_vgg = Model(inputs=base_model_vgg.input, outputs=predictions_vgg)
    return model_vgg

# ResNet50 
def create_resnet50_model():
    base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x_resnet = base_model_resnet.output
    x_resnet = GlobalAveragePooling2D()(x_resnet)
    x_resnet = Dense(1024, activation='relu')(x_resnet)
    predictions_resnet = Dense(1, activation='sigmoid')(x_resnet)
    model_resnet = Model(inputs=base_model_resnet.input, outputs=predictions_resnet)
    return model_resnet

# ResNet101
def create_resnet101_model():
    base_model_resnet = ResNet101(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x_resnet = base_model_resnet.output
    x_resnet = GlobalAveragePooling2D()(x_resnet)
    x_resnet = Dense(1024, activation='relu')(x_resnet)
    predictions_resnet = Dense(1, activation='sigmoid')(x_resnet)
    model_resnet = Model(inputs=base_model_resnet.input, outputs=predictions_resnet)
    return model_resnet

# function to train
def train_model(model, train_data, val_data, epochs=50, save_path='model.h5'):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=[
            LearningRateScheduler(lambda epoch: 0.001 if epoch < 10 else 0.0001),
            ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', mode='min', verbose=1)
        ]
    )
    return history

# Train and save the VGG16 
vgg_model = create_vgg_model()
vgg_history = train_model(vgg_model, train_generator, val_generator, save_path='vgg_model.h5')

# Train and save the ResNet50 
resnet50_model = create_resnet50_model()
resnet50_history = train_model(resnet_model, train_generator, val_generator, save_path='resnet50_model.h5')

# Train and save the ResNet101 
resnet101_model = create_resnet101_model()
resnet101_history = train_model(resnet101_model, train_generator, val_generator, save_path='resnet101_model.h5')

