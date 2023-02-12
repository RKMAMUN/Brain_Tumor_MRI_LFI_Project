# Brain_Tumor_MRI_LFI_Project

### Introduction
<p align="justify">Tumor genomic clusters refer to groups of genetic mutations that are frequently found together in tumors. These mutations can drive the growth and development of the cancer, and the clustering of certain mutations can provide insight into the underlying biology of the tumor and inform treatment decisions. In cancer genomics, the identification and analysis of tumor genomic clusters is an important tool for understanding the genetic changes that occur in cancer cells and for developing new treatments.</br> </p>

<p align="center"><img width="460" height="300" src="brain.jpg"></p>

In this project, segmentation masks for pictures of brain tumors are made using the U-Net Architecture and ResUnet

### Overview
- - - -
* Dataset
* Tools/IDE used
* Simple workflow
* Data visualization
* Methodology
* Evaluation and Result
* Challenges
* References

### Dataset Used
<p align="justify">The dataset consists of MRI scans of the brain, along with manual segmentations of the tumors created by experienced radiologists. The segmentations serve as ground truth annotations and are used to evaluate the performance of algorithms designed to segment LGGs in MRI scans. The dataset can be used to train and test machine learning models, such as Convolutional Neural Networks (CNNs), for the task of brain tumor segmentation.</p></br>
<p align="justify">The LGG Segmentation Dataset is a valuable resource for researchers and practitioners in the field of medical imaging, and its use can lead to the development of more accurate and efficient algorithms for the detection and diagnosis of brain tumors. </p>

<p float="left">
  <img src="images/original.jpg" width="370" height="250" />
  <img src="images/mask.jpg" width="370" height="250" /> 
</p>


### Tools/IDE used
<p align="justify">The project is developed in python using Jupyter notebook online and Kaggle GPU. we have used Kaggle GPU notebook online to show the flow of the application as it provides a block execution interface and each stage output can be shown effectively.</p>

Machine Learning and Python Packages:
NumPy,
pandas,
Matplotlib,
Seaborn,
scikit-learn,
cv2,
tensorflow

 ### Simple workflow
<p align="center"><img src="images/workflow.JPG"></p>


### Data visualization




### Challenges
<p align="justify">Image Variability: MRI scans can have significant variability in terms of contrast, intensity, and resolution. This can result in different appearances of the same type of tumor in different images, making it challenging to develop algorithms that can accurately</p>

<p align="justify">Complexity of the model: The complexity of the model being used for training also affects the training time. More complex models, such as deep CNNs with multiple layers and large numbers of parameters, require more time to train than simpler models. </p>

<p align="justify">Size of the dataset: The size of the dataset used for training directly affects the training time. Larger datasets require more time to train, as the algorithm needs to process more images. </p>

<p align="justify">Hardware: The training time can be significantly impacted by the hardware being used. Training on GPUs is typically faster than training on CPUs, and the training time can be further reduced by using multiple GPUs. </p>

### References
https://paperswithcode.com/paper/association-of-genomic-subtypes-of-lower
https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation
https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
https://iq.opengenus.org/resnet50-architecture/
https://medium.com/analytics-vidhya/brain-tumor-detection-and-segmentation-model-fc5dc952f6fe
https://www.analyticsvidhya.com/blog/2021/06/brain-tumor-detection-and-localization-using-deep-learning-part-1/
https://www.analyticsvidhya.com/blog/2021/06/brain-tumor-detection-and-localization-using-deep-learning-part-2/

