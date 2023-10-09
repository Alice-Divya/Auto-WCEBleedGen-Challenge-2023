#!/usr/bin/env python
# Import the reqouied Libraries
import os
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import KFold
from tensorflow.keras.applications import ResNet50  # Import Resnet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint


# Function to load and preprocess data
def load_and_preprocess_data(data_folder):
    image_data = []
    labels = []
    
    for category in os.listdir(data_folder):
        category_folder = os.path.join(data_folder, category)
        if os.path.isdir(category_folder):
            for subcategory in os.listdir(category_folder):
                subcategory_folder = os.path.join(category_folder, subcategory)
                if os.path.isdir(subcategory_folder):
                    for file in os.listdir(subcategory_folder):
                        if file.endswith(".png"):
                            img = tf.keras.preprocessing.image.load_img(
                                os.path.join(subcategory_folder, file),
                                target_size=(224, 224)  # Adjust target size for VGG16
                            )
                            img_array = tf.keras.preprocessing.image.img_to_array(img)
                            img_array = tf.keras.applications.vgg16.preprocess_input(img_array)  # Use VGG16 preprocessing
                            image_data.append(img_array)
                            labels.append(category)
    
    return np.array(image_data), labels


# Data set path
data_dir = "data_path"


data,labels = load_and_preprocess_data(data_dir)

#shuffling the data
X, labels = shuffle(data, labels, random_state=42)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)


# Function for learning rate based on the epochs
def learning_rate_schedule(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.0001

lr_scheduler = LearningRateScheduler(learning_rate_schedule)


# Function to Add early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True
)

#Loading the model with imagenet weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Adjust input shape for VGG16
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# K fold split for the data to be trained
i = 0 # initialising  the model with i to save with the same different name for each fold
kfold = KFold(n_splits=5, shuffle=False)
for train_indices, val_indices in kfold.split(X):
    X_train,X_val = X[train_indices],X[val_indices]
    y_train,y_val = y[train_indices],y[val_indices]
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #To save the model with best weights
    model_checkpoint = ModelCheckpoint(f'best_model_{i}.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1)
    history = model.fit(
        X_train, y_train,
        epochs=100, batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[lr_scheduler,early_stopping, model_checkpoint]
    )
    #Saving the  weights of the last moodel
    model.save(f'last_weight_{i}.h5')
    i+=1