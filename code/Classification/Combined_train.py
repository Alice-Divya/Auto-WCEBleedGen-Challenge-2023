#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50  # Import VGG16 and ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
import efficientnet.keras as efn
efficientnet.keras import EfficientNetB0

# Define the root directory path
root_folder = 'D:/JammuO'

# Define a function to load and preprocess data
def load_and_preprocess_data(data_folder, preprocessing_function):
    datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    generator = datagen.flow_from_directory(
        data_folder,
        target_size=(224, 224),  # Adjust target size for VGG16, ResNet50, and EfficientNetB0
        batch_size=32,
        class_mode='binary'
    )
    return generator

# Load and preprocess training data
train_generator = load_and_preprocess_data(os.path.join(root_folder, 'train'), tf.keras.applications.vgg16.preprocess_input)

# Load and preprocess validation data
val_generator = load_and_preprocess_data(os.path.join(root_folder, 'validation'), tf.keras.applications.vgg16.preprocess_input)

# Create a VGG16 model for feature extraction
def create_vgg_model():
    base_model_vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x_vgg = base_model_vgg.output
    x_vgg = GlobalAveragePooling2D()(x_vgg)
    x_vgg = Dense(1024, activation='relu')(x_vgg)
    predictions_vgg = Dense(1, activation='sigmoid')(x_vgg)
    model_vgg = Model(inputs=base_model_vgg.input, outputs=predictions_vgg)
    return model_vgg

# Create a ResNet50 model for feature extraction
def create_resnet_model():
    base_model_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x_resnet = base_model_resnet.output
    x_resnet = GlobalAveragePooling2D()(x_resnet)
    x_resnet = Dense(1024, activation='relu')(x_resnet)
    predictions_resnet = Dense(1, activation='sigmoid')(x_resnet)
    model_resnet = Model(inputs=base_model_resnet.input, outputs=predictions_resnet)
    return model_resnet

# Create an EfficientNetB0 model for feature extraction
def create_effnet_model():
    base_model_effnet = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
    x_effnet = base_model_effnet.output
    x_effnet = GlobalAveragePooling2D()(x_effnet)
    x_effnet = Dense(1024, activation='relu')(x_effnet)
    predictions_effnet = Dense(1, activation='sigmoid')(x_effnet)
    model_effnet = Model(inputs=base_model_effnet.input, outputs=predictions_effnet)
    return model_effnet

# Define a function to train a model
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

# Train and save the VGG16 model
vgg_model = create_vgg_model()
vgg_history = train_model(vgg_model, train_generator, val_generator, save_path='vgg_model.h5')

# Train and save the ResNet50 model
resnet_model = create_resnet_model()
resnet_history = train_model(resnet_model, train_generator, val_generator, save_path='resnet_model.h5')

# Train and save the EfficientNetB0 model
effnet_model = create_effnet_model()
effnet_history = train_model(effnet_model, train_generator, val_generator, save_path='effnet_model.h5')

