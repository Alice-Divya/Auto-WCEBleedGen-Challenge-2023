#!/usr/bin/env python
# coding: utf-8

# **Model loading and ploting validation images**

# In[9]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_flat = tf.keras.layers.Flatten()(y_true)
    y_pred_flat = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat)
    dice_coefficient = (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_flat) + tf.reduce_sum(y_pred_flat) + smooth)
    return 1.0 - dice_coefficient


def mean_iou(y_true, y_pred):
    intersection = tf.reduce_sum(tf.round(tf.clip(y_true * y_pred, 0, 1)))
    union = tf.reduce_sum(tf.round(tf.clip(y_true + y_pred, 0, 1)))
    return intersection / (union + tf.keras.backend.epsilon())

# Loading our saved bestmodel
model_path = '/content/drive/MyDrive/splitted datanew/bestresidual_model.h5'
model = tf.keras.models.load_model(model_path, custom_objects={'dice_loss': dice_loss, 'mean_iou': mean_iou})

batch_size = 8
#predictions on validation data
predictions = model.predict(X_valid, batch_size=batch_size)
num_samples = len(X_valid)

#plotting Original Image,Ground Truth Mask,Predicted Mask

plt.figure(figsize=(12, 4))
for i in range(num_samples):
    
    original_image = X_valid[i]
    ground_truth_mask = y_valid[i]
    predicted_mask = predictions[i]

    # OriginalImage
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    # Test Mask
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(ground_truth_mask), cmap='gray')
    plt.title('Ground Truth Mask')
    plt.axis('off')

    # Predicted Mask
    plt.subplot(1, 3, 3)
    plt.imshow(np.squeeze(predicted_mask), cmap='gray')
    plt.title('Predicted Mask')
    plt.axis('off')

    plt.show()
plt.tight_layout()
plt.show()


# **Saving predicted mask and ground truth mask**

# In[ ]:


import os
import matplotlib.pyplot as plt
import numpy as np

# saving predicted mask and groundtruth mask
predicted_mask_dir = '/content/drive/MyDrive/splitted datanew/predicted_masks'
ground_truth_mask_dir = '/content/drive/MyDrive/splitted datanew/ground_truth_masks'

os.makedirs(predicted_mask_dir, exist_ok=True)
os.makedirs(ground_truth_mask_dir, exist_ok=True)

for i in range(num_samples):
    ground_truth_mask = y_valid[i]
    predicted_mask = predictions[i]

    
    ground_truth_mask_filename = os.path.join(ground_truth_mask_dir, f'ground_truth_mask_{i}.png')
    plt.imshow(np.squeeze(ground_truth_mask), cmap='gray')
    plt.axis('off')
    plt.savefig(ground_truth_mask_filename)
    plt.clf()  

    
    predicted_mask_filename = os.path.join(predicted_mask_dir, f'predicted_mask_{i}.png')
    plt.imshow(np.squeeze(predicted_mask), cmap='gray')
    plt.axis('off')
    plt.savefig(predicted_mask_filename)
    plt.clf()

print("Predicted masks and ground truth masks saved successfully.")

