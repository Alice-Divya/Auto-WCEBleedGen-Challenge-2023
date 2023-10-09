# Auto-WCEBleedGen Challenge-2023
## Automatic Detection and Classification of Bleeding and Non-Bleeding frames in Wireless Capsule Endoscopy

 

We showcase our solution and subsequent progress for the [Auto-WCEBleedGen Challenge](https://misahub.in/CVIP/challenge.html). This challenge provides an unique opportunity to create, test, and evaluate cutting-edge Artificial Intelligence (AI) models for automatic detection and classification of bleeding and non-bleeding frames extracted from Wireless Capsule Endoscopy (WCE) videos. Our project make use of the dataset created by the Auto-WCEBleedGen Challenge Organizers. The training dataset encompasses a wide array of gastrointestinal bleeding scenarios, accompanied by medically validated binary masks and bounding boxes. Our objective is to facilitate comparisons with existing methods and enhance the interpretability and reproducibility of automated systems in bleeding detection using WCE data.

 

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
Gastrointestinal(GI) bleeding is a medical condition characterized by bleeding in the digestive tract, which circumscribes esophagus, stomach, small intestine, large intestine (colon), rectum, and anus. Wireless capsule endoscopy (WCE) is an efﬁcient tool to investigate GI tract disorders and perform painless imaging of the intestine([Figure-1](https://private-user-images.githubusercontent.com/146803475/272840458-4550a321-1ae0-47a8-99c7-4a96ba89e377.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY1MDg3MzcsIm5iZiI6MTY5NjUwODQzNywicGF0aCI6Ii8xNDY4MDM0NzUvMjcyODQwNDU4LTQ1NTBhMzIxLTFhZTAtNDdhOC05OWM3LTRhOTZiYTg5ZTM3Ny5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNVQxMjIwMzdaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1hOTQxYmU1MmEwZTc3N2RjMzkzYzdlNzMzNTcyZDhjNDgyODJhMzhiODM4ODliODZjNTNiZWE0NDFlYWYxM2IyJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.1z-miiJi9Tdu2Fd7LnCSbdaWS17tkow7njklUl_MJ6k)). It currently takes an experienced gastroenterologist approximately 2 with 3 hours to inspect the captured video by WCE of one patient frame-by-frame, which can result in human errors. Considering the poor ratio of patients to doctors in developing countries like India, the need for research and development of robust, interpretable, and generalized AI models has arisen. computer-aided classification and detection of bleeding and non-bleeding frames, gastroenterologists can reduce their workload and save their valuable time. Our project adopts a deep neural network approach and we concluded the model(1) for classification and model(2) for segmentation as best models according to its accuracy and IoU values respectively. The achieved accuracy of best model for classification is ___ and IoU of best model for segmentation is _____.

 




<p align="center"> 
<img width="524" alt="MicrosoftTeams-image (2)" src="https://github.com/Alice-Divya/Auto-WCEBleedGen-Challenge-2023/assets/146923115/955b40c3-cea0-4968-a653-75c7647692cb">
</p>

## Dataset
The Given training dataset consists of 2618 color images obtained from WCE. The images are in 24-bit PNG format, with 224 × 224 pixel resolution. The dataset is composed of two equal subsets, 1309 images as bleeding images and 1309 as non-bleeding images. Also it has Corresponding binary mask images for both subsets([Figure-2](https://private-user-images.githubusercontent.com/146803475/272868281-dd49c9f8-d830-4661-9382-9dc34e5ee7ff.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTEiLCJleHAiOjE2OTY1MDkyMTgsIm5iZiI6MTY5NjUwODkxOCwicGF0aCI6Ii8xNDY4MDM0NzUvMjcyODY4MjgxLWRkNDljOWY4LWQ4MzAtNDY2MS05MzgyLTlkYzM0ZTVlZTdmZi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBSVdOSllBWDRDU1ZFSDUzQSUyRjIwMjMxMDA1JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDIzMTAwNVQxMjI4MzhaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT00N2QxYjM2ZWY1NGNjMTA1ZDU1ODc2NmRhM2FhY2ExZjIwY2Q3Mjg4ODk1OGIxYmFlYzczNjc4OTNmN2FjMzI4JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.hgDs52trGd_1SZfOyqYy2P301pgBjkYh-uanrIgHwDQ)). Each subset is split into two parts, 1049 (80%) images for training and 260 (20%)images for validation. The bleeding subset is annotated by human expert and contains 1309 binary masks in PNG format of the same 224 x 224 pixel resolution. White pixels in the masks correspond to bleeding localization.

 

<img width="777" alt="image" src="https://github.com/NPCalicut/Blood/assets/146803475/dd49c9f8-d830-4661-9382-9dc34e5ee7ff">
