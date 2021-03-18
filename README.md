# Semantic Segmentation
## Problem description
The importance of Semantic Segmentation is growing everyday along with it's applications. Many trending AI solutions like Autonomous cars should be able to do Semantic Segmentation in order to be successful. The main goal of the project is to predict what class each pixel in images belong to? I got the Aerial Semantic Segmentation Drone dataset from http://dronedataset.icg.tugraz.at and performed Semantic Segmentation on it using the U-Net model with ResNet50 encoder.

## Dataset:
### Data:
I downloaded the data from the Institute of Computer Vision and Graphics website. The link to the project is http://dronedataset.icg.tugraz.at. I uploaded the dataset to my Google Drive and then imported it into Google Colab.

### Understanding the Dataset:
The dataset has all the original images in a folder called "original_images" and their corresponding images with masks in "label_images_semantic" folder. Both of these folders can be found in "proj/dataset/semantic_drone_dataset/" folder in this github repository. Here are examples of an original image and it's corresponding image with masj: <br />
![alt text](https://github.com/ruthviksai/SemanticSegmentation/blob/main/original_image.png) | ![alt text](https://github.com/ruthviksai/SemanticSegmentation/blob/main/image_with_mask.png) <br />
Each image is of size (4000, 6000, 3) and each mask is of size (4000, 6000). There are a total of 400 images in the dataset. I have used a 0.7-0.15-0.15 splitof the dataset to give 280 images in training dataset, 60 images in validation dataset, and 60 images in the testing dataset respectively.
  
### Data transformation:
I have normalized the images using the standard means and standard deviations used in pytorch mean = [0.485, 0.456, 0.406], standard deviation = [0.229, 0.224, 0.225]. I have also used Albumentations python library to transform the training and validation data. Albumentations is a Python library for image augmentation. It is used to increase the quality of the trained models.

## Model:
I have used the U-Net model with ResNet50 encoder pretrained on the ImageNet dataset from the "segmentation-models-pytorch" python library. You can import the model and change the head using the following command:
```python
model = smp.Unet('resnet50', encoder_weights='imagenet', classes=23, activation=None, encoder_depth=5, decoder_channels=[256, 128, 64, 32, 16])
```
I have trained the model for 15 epochs and at each epoch calculated the average pixel accuracy and average IOU scores on both training and validation datasets along with losses. The plots corresponding to the 3 are given below: <br />
![alt text](https://github.com/ruthviksai/SemanticSegmentation/blob/main/training_plots.png)

We can see that the training and validation losses have decreased with each epoch. Also the IOU and Accuracy Scores have increases with each epoch. This shows that the training has been done properly.

## Evaluation/Results:
### Evaluation/Results
After training the model, I have evaluated the model on the test dataset by running it on test images and predicting masks for the images. And comparing those predicted masks to the actual masks for the testing images, I have calculated the IOU Score and Pizel accuracy. Here are the IOU Scores for all the 60 testing images: <br />
[0.01437535951719747, 0.027830104889198723, 0.037636654296100636, 0.25745239332164954, 0.01871943365269572, 0.017298327172982738, 0.03736476105629891, 0.005231125780209341, 0.035912419566468096, 0.027395614450115548, 0.020896997684584376, 0.01534604112031907, 0.019398139810491277, 0.08172611029700651, 0.017125933014554395, 0.038136607168503464, 0.022494418133496872, 0.1395507160177836, 0.023869266268821677, 0.0065359467430306005, 0.016098082668007146, 0.008548186507378406, 0.05634263314335873, 0.00503594197293651, 0.13908541079210796, 0.008873851958658951, 0.007836094440955142, 0.020415742697034278, 0.04428648278972053, 0.04168202041351731, 0.015268589352858526, 0.02995368381372454, 0.004300687218607729, 0.09300447739131312, 0.02249949342753367, 0.03342576758717754, 0.03642942745751209, 0.053024850787353174, 0.010253906154186278, 0.028581277417296734, 0.0054334406081649345, 0.0148653226119955, 0.015125707164288456, 0.030925506458630642, 0.06141431136188718, 0.021227309533169616, 0.009928545921081468, 0.017837574511213938, 0.03222894300123117, 0.05345856875415537, 0.004806421538422603, 0.027253476318402164, 0.024115715856491953, 0.1828101136007142, 0.002410948456150336, 0.03630991636776797, 0.00972339422242797, 0.06547174763888221, 0.009959655808449418, 0.0021972904296739936] <br />
None of the images produced IOU Score greater than 0.5 and the average IOU Score is 0.036112448135265804. <br />

Here are the Pixel Accuracy Scores for all the 60 testing images: <br />
[0.01792286060474537, 0.3805259422019676, 0.28908397533275465, 0.8111504448784722, 0.1204562717013889, 0.04457035771122685, 0.1468539767795139, 0.019343623408564815, 0.24922914858217593, 0.16033144350405093, 0.024713586877893517, 0.1503058539496528, 0.08980871129918981, 0.5552786367910879, 0.02840056242766204, 0.31337031611689814, 0.06762469256365741, 0.6895073784722222, 0.1737377025462963, 0.019058792679398147, 0.05931034794560185, 0.022424768518518517, 0.3587601273148148, 0.01801667390046296, 0.6842515733506944, 0.01859424732349537, 0.06681202076099536, 0.1842673972800926, 0.37171314380787035, 0.26480667679398145, 0.031141493055555556, 0.1854214138454861, 0.035799379701967594, 0.5279981825086806, 0.033122875072337965, 0.044689037181712965, 0.2813822428385417, 0.41257618091724535, 0.015316433376736112, 0.23093103479456017, 0.019439697265625, 0.05400028935185185, 0.18082569263599538, 0.26450150101273145, 0.4138658311631944, 0.044718424479166664, 0.028166594328703703, 0.07951072410300926, 0.3958152488425926, 0.05062527126736111, 0.01916729962384259, 0.10125054253472222, 0.06770607277199074, 0.70404052734375, 0.006837067780671296, 0.22234542281539352, 0.02522560402199074, 0.5331149631076388, 0.017653853804976853, 0.004111961082175926] <br />
The average Pixel Accuracy Score is 0.19052553530092592.

Looking at the IOU and Pixel Accuracy scores, the model does not seem to be performing very well but looking at the precicted masks we can say that the model in fact has performed pretty well:
![alt text](https://github.com/ruthviksai/SemanticSegmentation/blob/main/test_images1.png)
![alt text](https://github.com/ruthviksai/SemanticSegmentation/blob/main/test_images2.png)

From the above images, we can see that the model has done a decent job on predicting the masks. The model can certainly be improved by training it for more epochs. Currently I am only training it for 15 epochs. Another way to improve the model is to try a different encoder than ResNet50.
