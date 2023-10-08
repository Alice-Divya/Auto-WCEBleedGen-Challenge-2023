#!/usr/bin/env python
# coding: utf-8

# **Prediction on Testdata 1**

# In[8]:


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

test_image_dir = '/content/drive/MyDrive/testdata1'

test_images = []
t1 = []

]
for filename in os.listdir(test_image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(test_image_dir, filename)
        img1 = plt.imread(image_path)
        image = cv2.imread(image_path) / 255.0 
        target_size = (224, 224)
        resized_image = cv2.resize(image, target_size)
        test_images.append(resized_image)
        t1.append(img1)

# numpy array conversion
test_images = np.array(test_images)


batch_size = 8

#predictions on the test images
num_images = len(test_images)
predicted_masks = []
for i in range(0, num_images, batch_size):
    batch = test_images[i:i + batch_size]
    batch_predictions = model.predict(batch)
    predicted_masks.extend(batch_predictions)


for i in range(len(test_images)):
    plt.figure(figsize=(15, 5))

    original_image = t1[i] 
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')


    predicted_mask = (predicted_masks[i].squeeze() * 255).astype(np.uint8) 
    predicted_mask_resized = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]))
    predicted_mask_resized_3channels = cv2.merge([predicted_mask_resized] * 3)

    plt.subplot(1, 3, 2) 
    plt.imshow(predicted_mask_resized_3channels, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')


    overlay = cv2.addWeighted((original_image * 255).astype(np.uint8), 0.7, predicted_mask_resized_3channels, 0.3, 0)
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')

    plt.show()


# **Prediction on Testdata2**

# In[10]:


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

test_image_dir = '/content/drive/MyDrive/testdata2'

test_images = []
t1 = []


for filename in os.listdir(test_image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):

        image_path = os.path.join(test_image_dir, filename)
        img1 = plt.imread(image_path)
        image = cv2.imread(image_path) / 255.0

        
        target_size = (224, 224)
        resized_image = cv2.resize(image, target_size)
        test_images.append(resized_image)
        t1.append(img1)


test_images = np.array(test_images)
batch_size = 8 


num_images = len(test_images)
predicted_masks = []
for i in range(0, num_images, batch_size):
    batch = test_images[i:i + batch_size]
    batch_predictions = model.predict(batch)
    predicted_masks.extend(batch_predictions)

    
for i in range(len(test_images)):
    plt.figure(figsize=(15, 5))


    original_image = t1[i]
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')


    predicted_mask = (predicted_masks[i].squeeze() * 255).astype(np.uint8) 
    predicted_mask_resized = cv2.resize(predicted_mask, (original_image.shape[1], original_image.shape[0]))
    predicted_mask_resized_3channels = cv2.merge([predicted_mask_resized] * 3)

    plt.subplot(1, 2, 2)  
    plt.imshow(predicted_mask_resized_3channels, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')



    plt.show()

