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


# In[5]:


# Define the root directory path
root_folder = 'D:/JammuO'


# In[6]:


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


# In[7]:


# Load and preprocess training data
X_train, y_train = load_and_preprocess_data(os.path.join(root_folder, 'train'))

# Load and preprocess validation data
X_val, y_val = load_and_preprocess_data(os.path.join(root_folder, 'validation'))


# In[8]:


# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)


# In[11]:


# Create a ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))  # Adjust input shape for ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)


# In[12]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[13]:


# Define a learning rate scheduler
def learning_rate_schedule(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.0001

lr_scheduler = LearningRateScheduler(learning_rate_schedule)


# In[15]:


# Save the best model during training
model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, save_weights_only=True, monitor='val_loss', mode='min', verbose=1)


# In[16]:


# Train the model on your training data with callbacks
history = model.fit(
    X_train, y_train_encoded,
    epochs=50, batch_size=32,
    validation_data=(X_val, y_val_encoded),
    callbacks=[lr_scheduler, model_checkpoint]
    
)


# In[17]:


# Evaluate the model using additional metrics
y_val_pred = model.predict(X_val)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)
report = classification_report(y_val_encoded, y_val_pred_binary, target_names=label_encoder.classes_)
print(report)


# In[18]:


# Save the trained model
model.save('Resnet50.h5')


# In[19]:


# Evaluate the model on the validation data
val_loss, val_accuracy = model.evaluate(X_val, y_val_encoded)

# Print the validation accuracy
print(f'Validation Accuracy: {val_accuracy:.4f}')


# In[22]:


from sklearn.metrics import precision_score

# Predict probabilities on the validation data
y_val_pred = model.predict(X_val)

# Convert probabilities to binary predictions (0 or 1)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)

# Calculate precision
precision = precision_score(y_val_encoded, y_val_pred_binary)

# Print the precision
print(f'Precision: {precision:.4f}')


# In[23]:


from sklearn.metrics import f1_score, recall_score

# Predict probabilities on the validation data
y_val_pred = model.predict(X_val)

# Convert probabilities to binary predictions (0 or 1)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)

# Calculate F1-score and recall
f1 = f1_score(y_val_encoded, y_val_pred_binary)
recall = recall_score(y_val_encoded, y_val_pred_binary)

# Print the F1-score and recall
print(f'F1-Score: {f1:.4f}')
print(f'Recall: {recall:.4f}')


# In[24]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Predict probabilities on the validation data
y_val_pred = model.predict(X_val)

# Convert probabilities to binary predictions (0 or 1)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)

# Calculate the confusion matrix
cm = confusion_matrix(y_val_encoded, y_val_pred_binary)

# Define class labels
class_labels = ['Non-Bleeding', 'Bleeding']

# Create a heatmap of the confusion matrix with labels
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




