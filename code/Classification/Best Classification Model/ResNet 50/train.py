#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50  # Import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.applications import ResNet101


# In[2]:


# Define the root directory path
root_folder = 'D:/JammuO'


# In[3]:


# Define a function to load and preprocess data
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
                               target_size=(224, 224)  # Adjust target size for ResNet50
                           )
                            img_array = tf.keras.preprocessing.image.img_to_array(img)
                            img_array = tf.keras.applications.resnet50.preprocess_input(img_array)  # Use ResNet50 preprocessing
                            image_data.append(img_array)
                            labels.append(category)

    return np.array(image_data), labels


# In[4]:


# Load and preprocess training data
X_train, y_train = load_and_preprocess_data(os.path.join(root_folder, 'train'))

# Load and preprocess validation data
X_val, y_val = load_and_preprocess_data(os.path.join(root_folder, 'validation'))


# In[5]:


# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)


# In[7]:


# Create a ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Adjust input shape for ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)


# In[8]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[9]:


# Define a learning rate scheduler
def learning_rate_schedule(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.0001

lr_scheduler = LearningRateScheduler(learning_rate_schedule)


# In[11]:


# Save the best model during training
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1)


# In[12]:


# Train the model on your training data with callbacks
history = model.fit(
    X_train, y_train_encoded,
    epochs=50, batch_size=32,
    validation_data=(X_val, y_val_encoded),
    callbacks=[lr_scheduler, model_checkpoint]
)


# In[13]:


# Evaluate the model using additional metrics
y_val_pred = model.predict(X_val)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)
report = classification_report(y_val_encoded, y_val_pred_binary, target_names=label_encoder.classes_)
print(report)


# In[14]:


# Save the trained model
model.save('Resnet50Enhanced.h5')

