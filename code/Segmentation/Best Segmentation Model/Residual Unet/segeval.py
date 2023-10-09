#!/usr/bin/env python
# coding: utf-8

# **Calculating Mean IOU of images**

# In[ ]:


import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def calculate_iou(predicted_image_path, ground_truth_image_path):
    # Load the predicted and ground truth images
    predicted_image = cv2.imread(predicted_image_path, 0)  # Load as grayscale
    ground_truth_image = cv2.imread(ground_truth_image_path, 0)  # Load as grayscale

    #making predicted mask and ground truth same size
    if predicted_image.shape != ground_truth_image.shape:
        predicted_image = cv2.resize(predicted_image, (ground_truth_image.shape[1], ground_truth_image.shape[0]))

    # binary conversion
    _, predicted_binary = cv2.threshold(predicted_image, 0, 255, cv2.THRESH_BINARY)
    _, ground_truth_binary = cv2.threshold(ground_truth_image, 0, 255, cv2.THRESH_BINARY)

    
    intersection = np.logical_and(predicted_binary, ground_truth_binary)
    union = np.logical_or(predicted_binary, ground_truth_binary)
    iou = np.sum(intersection) / np.sum(union)

    return iou

def calculate_average_iou(predicted_folder, ground_truth_folder):
    predicted_images = sorted(os.listdir(predicted_folder))
    ground_truth_images = sorted(os.listdir(ground_truth_folder))

    total_iou = 0.0
    ious = []

    for predicted_image, ground_truth_image in zip(predicted_images, ground_truth_images):
        
        predicted_image_modified_path = os.path.join(predicted_folder, predicted_image)
        ground_truth_image_modified_path = os.path.join(ground_truth_folder, ground_truth_image)

        iou = calculate_iou(predicted_image_modified_path, ground_truth_image_modified_path)
        total_iou += iou
        ious.append(iou)

    average_iou = total_iou / len(predicted_images)

    return average_iou, ious

# folder path
predicted_folder = '/content/drive/MyDrive/splitted datanew/predicted_masks'
ground_truth_folder = '/content/drive/MyDrive/splitted datanew/ground_truth_masks'


mean_iou, ious = calculate_average_iou(predicted_folder, ground_truth_folder)


print("Mean IoU:", mean_iou)

# plotting IoU value
x = np.arange(len(ious))
plt.plot(x, ious)


plt.xlabel('Image Index')
plt.ylabel('IoU')
plt.title('IoU Values')
plt.show()


# **Average precision,Mean Average precision**

# In[ ]:


def calculate_precision_recall(predicted_folder, ground_truth_folder, threshold=0.5):
    predicted_images = sorted(os.listdir(predicted_folder))
    ground_truth_images = sorted(os.listdir(ground_truth_folder))

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for predicted_image, ground_truth_image in zip(predicted_images, ground_truth_images):
        predicted_image_modified_path = os.path.join(predicted_folder, predicted_image)
        ground_truth_image_modified_path = os.path.join(ground_truth_folder, ground_truth_image)

        iou = calculate_iou(predicted_image_modified_path, ground_truth_image_modified_path)
        if iou >= threshold:
            true_positives += 1
        else:
            false_positives += 1
            false_negatives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    return precision, recall

def calculate_average_precision(predicted_folder, ground_truth_folder, thresholds=np.arange(0.1, 1.0, 0.1)):
    average_precisions = []

    for threshold in thresholds:
        precision, recall = calculate_precision_recall(predicted_folder, ground_truth_folder, threshold)
        average_precisions.append(precision)

    average_precision = np.mean(average_precisions)
    return average_precision

def calculate_mean_average_precision(predicted_folder, ground_truth_folder):
    mean_average_precision = calculate_average_precision(predicted_folder, ground_truth_folder)
    return mean_average_precision

#folder path
predicted_folder = '/content/drive/MyDrive/splitted datanew/predicted_masks'
ground_truth_folder = '/content/drive/MyDrive/splitted datanew/ground_truth_masks'

#calling functions
mean_average_precision = calculate_mean_average_precision(predicted_folder, ground_truth_folder)
average_precision=calculate_average_precision(predicted_folder, ground_truth_folder)
print("Mean Average Precision (mAP):", mean_average_precision)
print(" Average Precision (AP):",average_precision)


