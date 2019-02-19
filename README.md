# Building-Extraction-in-Very-High-Resolution-Imagery-by-Dense-Attention-Networks
Python implementation of Convolutional Neural Network (CNN) used in paper

This repository includes functions to preprocess the input images and their respective polygons so as to create the input image patches 
and mask patches to be used for model training. The CNN used here is the Dense Attention Network (DAN) implemented in the paper 
'Building Extraction in Very High Resolution Imagery by Dense Attention Networks' by Yang H., Wu P., Yao X., Wu Y., Wang B., Xu Y. (2018)

The main differences between the implementations in the paper and the implementation in this repository is as follows:

- Sigmoid layer is used as the last layer instead of the softmax layer, in consideration of the fact that this is a binary classification problem
- Simple data augmentation is used to improve the rotation - invariance of target recognition by the DAN model

Requirements:

- cv2
- glob
- json
- numpy
- keras (Tensorflow backend)
- gdal
- rasterio
