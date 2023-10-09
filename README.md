# Auto-WCEBleedGen Challenge-2023
## Automatic Detection and Classification of Bleeding and Non-Bleeding frames in Wireless Capsule Endoscopy

We showcase our solution and subsequent progress for the[Auto-WCEBleedGen Challenge](https://misahub.in/CVIP/challenge.html). This challenge provides an unique opportunity to create, test, and evaluate cutting-edge Artificial Intelligence (AI) models for automatic detection and classification of bleeding and non-bleeding frames extracted from Wireless Capsule Endoscopy (WCE) videos. Our project make us the dataset created by the Auto-WCEBleedGen Challenge Organizers. The training dataset encompasses a wide array of gastrointestinal bleeding scenarios, accompanied by medically validated binary masks and bounding boxes. Our objective is to facilitate comparisons with existing methods and enhance the interpretability and reproducibility of automated systems in bleeding detection using WCE data.   

### Contents
* [Team members](https://github.com/NPCalicut/Blood/blob/Raghul_branch_1/README.md#team-members)
* [Overview](https://github.com/NPCalicut/Blood/blob/Raghul_branch_1/README.md#overview)
* [Data](https://github.com/NPCalicut/Blood/blob/Raghul_branch_1/README.md#data)
* [Method](https://github.com/NPCalicut/Blood/blob/Raghul_branch_1/README.md#models)
* [Training](https://github.com/NPCalicut/Blood/blob/Raghul_branch_1/README.md#training)
* [Results](https://github.com/NPCalicut/Blood/blob/Raghul_branch_1/README.md#results)
* [How to run](https://github.com/NPCalicut/Blood/blob/priya_branch_1/README.md#how-to-run)

## Team members
- Dr. Kalpana George
- Abhiram A P
- Alice Divya Nelson
- Harishma N

## Overview
Gastrointestinal (GI) bleeding is a medical condition characterized by bleeding in the digestive tract, which circumscribes esophagus, stomach, small intestine, large intestine (colon), rectum, and anus. Wireless capsule endoscopy (WCE) is an efﬁcient tool to investigate GI tract disorders and perform painless imaging of the intestine (Figure-1). For an experienced gastroenterologist, it is estimated to take atleast 2 to 3 hours for the inspection of the WCE captured video of a patient. This tedious process of frame-by-frame analysis can result in human errors. In a developing country like India, there is an increasing demand for the research and development of robust, interpretable, and generalized AI models to assist the doctors to address the challenges faced in reaching a conclusion on the WCE reports of increasing patients. Through the help of computer-aided classification and detection of bleeding and non-bleeding frames, gastroenterologists can reduce their workload and save their valuable time. Our project focuses on a deep neural network approach, and through the evaluation results obtained, the model Resnet50 and ResUnet is finalized as the best models for classification and segmentation respectively. The above conclusion of the models was specifically based on the model's accuracy metrics for classification and IoU (Intersection Over Union) metrics for segmentation, along with its prediction images. The achieved accuracy of the classification best model is 0.983 and the IoU value for the segmentation best model is 0.938. 

<img width="261" alt="image" src="https://github.com/NPCalicut/Blood/assets/146803475/4550a321-1ae0-47a8-99c7-4a96ba89e377">

## Dataset
The given training dataset consists of 2618 color images obtained from WCE. The images are in 24-bit PNG format, with 224 × 224 pixel resolution. The dataset is composed of two equal subsets, 1309 images as bleeding images and 1309 as non-bleeding images. Also it has Corresponding binary mask images for both subsets[Figure-2](https://private-user-images.githubusercontent.com/146803475/272868281-dd49c9f8-d830-4661-9382-9dc34e5ee7ff.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY1MDkyMTgsIm5iZiI6MTY5NjUwODkxOCwicGF0aCI6Ii8xNDY4MDM0NzUvMjcyODY4MjgxLWRkNDljOWY4LWQ4MzAtNDY2MS05MzgyLTlkYzM0ZTVlZTdmZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNVQxMjI4MzhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00N2QxYjM2ZWY1NGNjMTA1ZDU1ODc2NmRhM2FhY2ExZjIwY2Q3Mjg4ODk1OGIxYmFlYzczNjc4OTNmN2FjMzI4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.hgDs52trGd_1SZfOyqYy2P301pgBjkYh-uanrIgHwDQ)). Each subset is split into two parts, 1049 (80%) images for training and 260 (20%)images for validation. The bleeding subset is annotated by human expert and contains 1309 binary masks in PNG format of the same 224 x 224 pixel resolution. White pixels in the masks correspond to bleeding localization.
<p align="center"> 
<img width="777" alt="image" src="https://github.com/NPCalicut/Blood/assets/146803475/dd49c9f8-d830-4661-9382-9dc34e5ee7ff">
</p>


## Method
### Analysing pixel intensity of bleeding and non-bleeding 
R, G, and B are the three color channels used in RGB pictures. The R channel is crucial for distinguishing between bleeding and non-bleeding pixels because bleeding typically manifests as red colors. Here, we plot the R, G, and B intensity histograms using training images with bleed and non-bleed regions shown in the [Figure-3](https://private-user-images.githubusercontent.com/146803475/272866245-7c36ffc9-b93f-4724-8e9b-c414bbfe41a6.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY1MDkzMTUsIm5iZiI6MTY5NjUwOTAxNSwicGF0aCI6Ii8xNDY4MDM0NzUvMjcyODY2MjQ1LTdjMzZmZmM5LWI5M2YtNDcyNC04ZTliLWM0MTRiYmZlNDFhNi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNVQxMjMwMTVaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02NzQwYzJkZmYxYWM3YzMxNTUzNmQ4YjUyZDk3NWExNWFiYzg4MmExNGYxYTE3Yzk1OGUwOWFlMmYyODQ3YzYzJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.JHgQRmwqtdeqNPJzv3_kRtCammnAzIrLRhuAMXzUIDo). We observed that while the G and B intensities varied considerably, the R intensities were comparable for both types of pixels. Green and blue intensity overlap was still too great to distinguish between bleeding and non-bleeding pixels of images. Therefore, classifying bleeding and non-bleeding images would be difficult using traditional machine learning algorithms. We therefore provide deep learning models here, which we utilized for classification and detection of bleeding images.

<img width="915" alt="image" src="https://github.com/NPCalicut/Blood/assets/146803475/7c36ffc9-b93f-4724-8e9b-c414bbfe41a6">


In light of deep learning techniques for classification, we initially tried a simple 5-layer scratch CNN model. Because of the unsatisfactory outcomes of this model, we switched to more complicated models.  The different deep architectures that we evaluated for classification are **VGG16, Resnet50 and EfficientnetB0**. Our analysis showed that the accuracy levels of VGG16 and EfficientnetB0 were comparable. But comparing predictions of test data, Resnet50 shows better result than VGG16 and EfficientnetB0, which prompted us to choose the *Resnet50* model as our chosen best classification model.


For segmentation, we tried the different models: **YOLOV8, Residual-Unet and Attention U-net**. We employ pre-trained encoders across all networks, which is an enhancement over regular U-Net. A U-Net-like architecture called Attention U-net  employs relatively straightforward pre-trained Imagenet networks as an encoder.
The Residual-Unet is an scratch model and additionally using a pre-trained network, YOLOV8 is a cutting-edge technique that also performed well. When comparing results, as best model *Residual-Unet* outperformed other models.


## Training
**Classification:**

The training dataset comprises 2618 color images and it is split in an 80:20 ratio for training and validation, respectively.
In 5-fold cross-validation, the data is divided into five equal-sized subsets or folds. The training process starts by training the model on four folds or subsets of the data, while the remaining fold acts as the validation set.The process described in [Figure 4(a)](https://github.com/NPCalicut/Blood/assets/146803475/cb634616-012a-4969-8503-274ddd02a4007) involves repeating the procedure five times, where each fold is used as the validation set once.

<p align="center">
  <img src="https://github.com/NPCalicut/Blood/assets/146803475/cb634616-012a-4969-8503-274ddd02a4007" alt="c1">
</p>



In summary, the training phase involves feeding the data into the CNN (Convolutional Neural Network) architecture, adjusting the model's parameters (weights and biases) based on the input data, and optimizing the model's performance. The validation phase involves assessing the performance of the trained model on a separate dataset that was not used during training, providing an indication of the model's generalization ability. After training, the best model was saved and select that model.

After evaluating different convolutional neural network (CNN) models, including the EfficientNetB0, VGG16, and ResNet50, it was concluded that Resnet50 demonstrated the best performance for prediction. That is highlighted in  [Figure 4(a)](https://private-user-images.githubusercontent.com/146803475/272896840-9f5b728b-405d-43b7-9db0-9dab1cfb7f7e.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY1NjYwNjksIm5iZiI6MTY5NjU2NTc2OSwicGF0aCI6Ii8xNDY4MDM0NzUvMjcyODk2ODQwLTlmNWI3MjhiLTQwNWQtNDNiNy05ZGIwLTlkYWIxY2ZiN2Y3ZS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNlQwNDE2MDlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1lN2NhYjc1YjBmN2RkZWNlZDhmM2QzNTE2MWQyNTkzN2RmYzllMzQ3MjZmN2ZhZDMyM2Q4MTJjYTg5NzRiZDBhJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.FdSDWv5Rng1l0OlKjFKVl2QUdpdkaJrQ2XE7o-20ysM).

According to [Figure 4(b)](https://private-user-images.githubusercontent.com/146803475/272897479-9669b100-c102-4bf5-8cab-1517d5895fa3.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY1NjYzMDksIm5iZiI6MTY5NjU2NjAwOSwicGF0aCI6Ii8xNDY4MDM0NzUvMjcyODk3NDc5LTk2NjliMTAwLWMxMDItNGJmNS04Y2FiLTE1MTdkNTg5NWZhMy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNlQwNDIwMDlaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02ZDc2YWE4MGMzOTlmY2JhMGU5ZTUyMGM5ZGUyMGZmMjY3ZmNlMTEzMmY5ZWJmZmE2NGE5YWJiOGE0YjQ4MWZkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.U1ec4wZlP8qGQ2HKVZaFQtgM1q-L5Gsh-N3P7opVpGs), the chosen model (Resnet50) is employed to classify test images as either bleeding or non-bleeding. To accomplish this, the test images is fed into the model for classification.


<p align="center">
  <img width =600 src="https://github.com/NPCalicut/Blood/assets/146803475/9669b100-c102-4bf5-8cab-1517d5895fa3" alt="clsts1">
</p>


 **Segmentation:**
 
During the training phase of the segmentation process, we experimented with different segmentation models including Residual-Unet, YOLOV8, and Attention U-net. These models were trained using annotated images and their corresponding ground truth data at the pixel level. 
<p align="center">
  <img src="https://github.com/NPCalicut/Blood/assets/146803475/148ee347-8697-4189-ba6c-394bcd5dd415" alt="s2">
</p>


Evaluation was conducted to assess their performance, and it was found that the Residual-Unet and YOLOV8 models yielded the most favourable results. This model is highlighted in [Figure 4(c)](https://private-user-images.githubusercontent.com/146803475/272869458-33fe57e8-a1ea-45fd-a6fe-cde8209526da.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY1NjYzODEsIm5iZiI6MTY5NjU2NjA4MSwicGF0aCI6Ii8xNDY4MDM0NzUvMjcyODY5NDU4LTMzZmU1N2U4LWExZWEtNDVmZC1hNmZlLWNkZTgyMDk1MjZkYS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNlQwNDIxMjFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mMDA4Y2IyNTJiZTNjY2Y0MTA2MDkyMTcwOGI5ZGQ4NmRkMmM1ZjE5ODRiZDU3ODBhNmI0YmFhOWUxZDZhN2U5JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.eGW4a0zv1QvjyUNbSWF5QOxg1AIiJdrAS6pmwgYqWnQ).

<p align="center">
  <img width =600 src="https://github.com/NPCalicut/Blood/assets/146803475/7f581e3d-39f7-4ea3-bdc4-4e0fba94d2aa" alt="Screenshot 2023-10-05 164823">
</p>

[Figure 4(d)](https://private-user-images.githubusercontent.com/146803475/272887275-7f581e3d-39f7-4ea3-bdc4-4e0fba94d2aa.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY1NjY0NDEsIm5iZiI6MTY5NjU2NjE0MSwicGF0aCI6Ii8xNDY4MDM0NzUvMjcyODg3Mjc1LTdmNTgxZTNkLTM5ZjctNGVhMy1iZGM0LTRlMGZiYTk0ZDJhYS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNlQwNDIyMjFaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05MDJhNDc2ZWQ0OGUyMTg4MGZkNmUxMmEwYzgxNzk1MTY4ODE3Y2RhZTUwMGE1ZTc5NTZhZDU1MDJjMjE0OGU1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.HIlNCwDfXmdLMPMf0rLObx8byiuS1cSXCnp6GUruMBE) illustrates the testing phase, where the chosen Residual-Unet model is employed to process the testing image, effectively identifying and detecting the presence of bleeding zones.

## Results

### 1. 5-Fold cross validation results for classification: 

Note: Here we put best accuracy of 5-fold cross-validation results  

**Table 1.1 :** Validation results of different classification models: 

| Model | Accuracy | Recall | F1-Score |
| --- | --- | --- | --- |
| VGG16 | 0.99 | 0.99 | 0.99 |
| Resnet50 |0.9828  |0.9885 |0.9829 |
| Resnet 101 | 0.962 |0.966 | 0.952|

According to **Table 1.1**, VGG16 achieved good accuracy, recall, and F1-scores. However, ResNet50 outperformed both VGG16 and ResNet101 in terms of prediction, with an accuracy of 0.9828, recall of 0.9885, and F1-score of 0.9829. Although ResNet101 attained flawless evaluation scores, ResNet50 is considered the best model out of five cross validation in terms of accuracy, recall and f1 score for classification tasks due to its superior performance compared to VGG16 and ResNet101. 


 **Screenshot of classification results:**

 
**1- Predicted Bleed class images from validation dataset**

<img width="862" alt="image" src="https://github.com/NPCalicut/Blood/assets/146803475/3f6e14ce-001b-42e5-8f34-6ee815aad139">


**2- Predicted Bleed class images from test dataset-1**

<img width="887" alt="image" src="https://github.com/NPCalicut/Blood/assets/146803475/8d281143-da63-4c3d-b6d6-e08eedc2e73b">

**3- Predicted Bleed class images from test dataset-2**

<img width="887" alt="image" src="https://github.com/NPCalicut/Blood/assets/146803475/8d281143-da63-4c3d-b6d6-e08eedc2e73b">

### 2. Validation results for segmentation:   

**Table 2.1 :** Validation results of different segmentation models: 

| Model | Accuracy | Recall | F1-Score |
| --- | --- | --- | --- |
| ResUnet | 0.9791 | 0.9791 | 0.938 |
| YOLOV8 |0.88  |0.88 |0.63 |
| Attention Unet | 0.272 |0.272 | 0.378|

 **Table 2.1** indicates that Residual-Unet outperforms the other two models in terms of average precision (AP) and mean average precision (MAP). The intersection over union (IOU) score for Residual-Unet is likewise comparatively high. Residual-Unet is therefore the most effective model out of the three for the segmentation task. 


 **Screenshot of segmentation results:**

 
**1-  Predictions of segmentation in validation dataset**

<img width="862" alt="image" src="https://github.com/NPCalicut/Blood/assets/146803475/3f6e14ce-001b-42e5-8f34-6ee815aad139">


**2- Predicted Bleed zone images from test dataset-1**

<img width="887" alt="image" src="https://github.com/NPCalicut/Blood/assets/146803475/8d281143-da63-4c3d-b6d6-e08eedc2e73b">

**3- Predicted Bleed zone images from test dataset-2**

<img width="887" alt="image" src="https://github.com/NPCalicut/Blood/assets/146803475/8d281143-da63-4c3d-b6d6-e08eedc2e73b">


## Complexity analysis of models

### Classification models:

| Model | size | Total parameters | Traininable parameters | No of layers | Time (ms) inference step (GPU) |
| --- | --- | --- | --- | --- | --- |
| ResUnet | 0.9791 | 0.9791 | 0.938 |  |  |
| YOLOV8 |0.88  |0.88 |0.63 |  |  |
| Attention Unet | 0.272 |0.272 | 0.378|  |  |

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 























**2.2.1 Predictions of segmentation in validation dataset**



**2.2.2 Predictions of segmentation in test dataset**

### 3. interpretability plot:

**CAMs**: *Class Activation Maps*, or *CAMs*, offer a means to see which pixels in an image most strongly influence how the model classifies it. In other words, a CAM shows how "important" each pixel in an input image is for a certain classification.

**Note:** Here, we used the Resnet50 model, which is our best model, to produce the CAMs plot.

**3.1.1 CAMs plot for validation dataset**



**3.1.2 CAMs plot of predictions using test dataset**


## How to run

The dataset is organized in the folloing way:
####
**classification** 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
**segmentation**

    ├── data                                                  ├── data
    │   ├── train                                             │   ├── train
    │   │     ├── Bleed                                       │   │     ├── Bleed 
    │   │     └── Nonbleed                                    │   │     │      ├── Images
    │   └── validation                                        │   │     │      └── Masks
    │           ├── Bleed                                     │   │     └── Nonbleed
    │           └── Nonbleed                                  │   │            ├── Images
    ├── test                                                  │   │            └── Masks
    │    ├── Testdata1                                        │   └── validation
    │    └── Testdata2                                        │         ├── Bleed
                                                              │         │     ├── Images
                                                              │         │     └── Masks
                                                              │         └── Nonbleed
                                                              │               ├── Images
                                                              │               └── Masks
                                                              └── test
                                                                    ├── Testdata1
                                                                    └── Testdata2

The training dataset contains two sets of images: one set with bleeding and the other set without bleeding. The entire training data is split with 80% allocated for training and 20% for validation. For training the dataset, we split it into five folds for five-fold cross-validation to perform classification.                                                                    
In order to perform segmentation, we utilized the WCEBleedGen dataset, which was also split in an 80:20 ratio for training and validation. The training dataset consists of two separate sets: one set containing bleeding images and their corresponding masks, and another set containing non-bleeding images and their corresponding masks. Similarly, the validation set follows the same format, with separate sets for bleeding and non-bleeding images along with their corresponding masks.

We used the Auto-WCEBleedGen Challenge Test Dataset for testing, which includes  tasetdata1 and testdata2 and only contain images. Thise dataset were used to assess model predictions.
1. Training

The main file that is used to train all models -  ``train.py``. Running ``python train.py --help`` will return set of all possible input parameters.
To train all models we used the folloing bash script (batch size was chosen depending on how many samples fit into the GPU RAM, limit was adjusted accordingly to keep the same number of updates for every network)::




