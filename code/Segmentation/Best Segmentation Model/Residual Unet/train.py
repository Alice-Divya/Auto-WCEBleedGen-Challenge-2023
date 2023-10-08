#!/usr/bin/env python
# coding: utf-8

# **Residual Unet**

# **Data loading**

# In[6]:


import os
import cv2
import numpy as np

# loading images and mask
def load_images_from_directory(image_directory, mask_directory, input_size):
    images = []
    masks = []

    for filename in os.listdir(image_directory):
        if filename.endswith(".png"):
            img_path = os.path.join(image_directory, filename)
            mask_path = os.path.join(mask_directory, filename)  

            img = cv2.imread(img_path)
            img = cv2.resize(img, (input_size[0], input_size[1]))
            img = img.astype(np.float32) / 255.0
            images.append(img)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (input_size[0], input_size[1]))
            mask = mask.astype(np.float32) / 255.0
            mask = np.expand_dims(mask, axis=-1)  
            masks.append(mask)

    return np.array(images), np.array(masks)

#  input size taken as(224,224)
input_size = (224,224 ,3)


train_image_data_dir = '/content/drive/MyDrive/splitted datanew/splitted data/train/images_bleed'
train_mask_data_dir = '/content/drive/MyDrive/splitted datanew/splitted data/train/masks_bleed'
valid_image_data_dir = '/content/drive/MyDrive/splitted datanew/splitted data/validation/images_bleed'
valid_mask_data_dir = '/content/drive/MyDrive/splitted datanew/splitted data/validation/masks_bleed'

X_train, y_train = load_images_from_directory(train_image_data_dir, train_mask_data_dir, input_size)
X_valid, y_valid = load_images_from_directory(valid_image_data_dir, valid_mask_data_dir, input_size)


# In[ ]:


print("Shape of X_train (input images):", X_train.shape)
print("Shape of y_train (masks):", y_train.shape)
print("Shape of X_valid (input images) :", X_valid.shape)
print("Shape of y_valid (masks):", y_valid.shape)


# **Metrics**

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D, Conv2DTranspose, concatenate


# code for Dice Coefficient ,Dice Coefficient Loss function, F1 Score function,Intersection over Union (IoU)

def dice_coeff(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)


def dice_loss(y_true, y_pred):
    return 1 - dice_coeff(y_true, y_pred)


def f1score(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    y_true = tf.keras.backend.cast(y_true, 'float32')
    y_pred = tf.keras.backend.cast(tf.keras.backend.round(y_pred), 'float32')
    tp = tf.keras.backend.sum(y_true * y_pred)
    fp = tf.keras.backend.sum(1 - y_true * y_pred)
    fn = tf.keras.backend.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

def mean_iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return tf.reduce_mean((intersection + 1e-15) / (union + 1e-15))


# **Model**

# In[1]:


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

input_size = (224, 224, 3)

#Residual  block
def residual_block(x, filters, kernel_size=3, stride=1):
   
    shortcut = x
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same', activation='relu')(x)
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same', activation='relu')(x)
    x = Concatenate()([x, shortcut])
    return x

# Residual U-Net model
def residual_unet(input_shape, num_classes=1):
    inputs = Input(input_shape)

    # Encoder
    conv1 = Conv2D(64, 3, padding='same', activation='relu')(inputs)
    conv1 = residual_block(conv1, 64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, padding='same', activation='relu')(pool1)
    conv2 = residual_block(conv2, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bridge
    conv3 = Conv2D(256, 3, padding='same', activation='relu')(pool2)
    conv3 = residual_block(conv3, 256)

    # Decoder
    up4 = UpSampling2D(size=(2, 2))(conv3)
    up4 = Conv2D(128, 2, padding='same', activation='relu')(up4)
    up4 = Concatenate()([up4, conv2])
    up4 = residual_block(up4, 128)

    up5 = UpSampling2D(size=(2, 2))(up4)
    up5 = Conv2D(64, 2, padding='same', activation='relu')(up5)
    up5 = Concatenate()([up5, conv1])
    up5 = residual_block(up5, 64)

    # Output
    output = Conv2D(num_classes, kernel_size=1, activation='sigmoid')(up5)  # Assuming binary segmentation, change num_classes for multiclass
    model = Model(inputs=inputs, outputs=output)
    return model

model = residual_unet(input_size)
model.summary()


# **Training**

# In[ ]:


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

# Compile the model
model = residual_unet(input_shape=input_size, num_classes=1)
model.compile(optimizer=Adam(learning_rate=1e-4), loss=dice_loss, metrics=[mean_iou, 'accuracy'])

# using checkpoint callback function for model saving
checkpoint_path = '/content/drive/MyDrive/splitted datanew/bestresidual_model.h5'
checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

# model training
history = model.fit(
    X_train, y_train,
    batch_size=8,
    validation_data=(X_valid, y_valid),
    epochs=50,
    callbacks=[checkpoint_callback]
)



#plotting metric curves

plt.figure(figsize=(16, 4))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(history.history['mean_iou'], label='Training Mean IoU')
plt.plot(history.history['val_mean_iou'], label='Validation Mean IoU')
plt.xlabel('Epochs')
plt.ylabel('Mean IoU')
plt.legend()

plt.show()

