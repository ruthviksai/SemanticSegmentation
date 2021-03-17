# Semantic Segmentation
## Problem description
The importance of Semantic Segmentation is growing everyday along with it's applications. Many trending AI solutions like Autonomous cars should be able to do Semantic Segmentation in order to be successful. The main goal of the project is to predict what class each pixel in images belong to? I got the Aerial Semantic Segmentation Drone dataset from http://dronedataset.icg.tugraz.at and performed Semantic Segmentation on it using the U-Net model with ResNet50 encoder.

## Dataset:
### Data:
I downloaded the data from the Institute of Computer Vision and Graphics website. The link to the project is http://dronedataset.icg.tugraz.at. I uploaded the dataset to my Google Drive and then imported it into Google Colab.

### Understanding the Dataset:
The dataset has all the original images in a folder called "original_images" and their corresponding images with masks in "label_images_semantic" folder. Both of these folders can be found in "proj/dataset/semantic_drone_dataset/" folder in this github repository. Here are examples of an original image and it's corresponding image with masj: <br />
![alt text](https://github.com/ruthviksai/SemanticSegmentation/blob/main/original_image.png) ![alt text](https://github.com/ruthviksai/SemanticSegmentation/blob/main/image_with_mask.png)
Each image is of size (4000, 6000, 3) and each mask is of size (4000, 6000). There are a total of 400 images in the dataset. I have used a 0.7-0.15-0.15 splitof the dataset to give 280 images in training dataset, 60 images in validation dataset, and 60 images in the testing dataset respectively.
  
### Data transformation:
I have normalized the images using the standard means and standard deviations used in pytorch mean = [0.485, 0.456, 0.406], standard deviation = [0.229, 0.224, 0.225]. I have also used Albumentations python library to transform the training and validation data. Albumentations is a Python library for image augmentation. It is used to increase the quality of the trained models.
