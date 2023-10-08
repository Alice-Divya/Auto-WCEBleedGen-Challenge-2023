#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os 
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the pre-trained ResNet50 model

model = tf.keras.models.load_model(r"C:\Users\36108\Downloads\best_model_4.h5")

# Define class labels for binary classification with reversed order

class_labels = ["Bleeding", "Non-Bleeding"]

# Define the paths to the test image folders

test_dataset2_folder = r"C:\Users\36108\Desktop\Auto-WCEBleedGen Challenge Test Dataset\Auto-WCEBleedGen Challenge Test Dataset\Test Dataset 2"
test_dataset1_folder = r"C:\Users\36108\Desktop\Auto-WCEBleedGen Challenge Test Dataset\Auto-WCEBleedGen Challenge Test Dataset\Test Dataset 1"

# Function to predict a single image

def predict_single_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    class_index = 0 if predictions[0] <= 0.5 else 1
    class_label = class_labels[class_index]

    confidence = predictions[0]

    return class_label, confidence

# Function to predict images in a folder and return as a DataFrame

def predict_images_in_folder_to_dataframe(folder_path):
    image_paths = []
    class_labels = []
    confidences = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            class_label, confidence = predict_single_image(img_path)

            image_paths.append(filename)
            class_labels.append(class_label)
            confidences.append(f'{confidence[0]:5f}')

        df = pd.DataFrame({
        "Image Path": image_paths,
        "Predicted Class": class_labels,
        "Confidence": confidences
        })

    return df

# Predict images in Test Dataset 2 and store the results in a DataFrame

predictions_dataset1_df = predict_images_in_folder_to_dataframe(test_dataset1_folder)
predictions_dataset2_df = predict_images_in_folder_to_dataframe(test_dataset2_folder)

# Specify the Excel file path where you want to save the predictions

excel_file = 'ResNet50_model_Predictions.xlsx'

# Write to Excel sheet with two separate sheets for each set of predictions

with pd.ExcelWriter(excel_file) as writer:

    predictions_dataset1_df.to_excel(writer, sheet_name='Set1_Predictions', index=False)

    predictions_dataset2_df.to_excel(writer, sheet_name='Set2_Predictions', index=False)

print('Predictions written to', excel_file)

