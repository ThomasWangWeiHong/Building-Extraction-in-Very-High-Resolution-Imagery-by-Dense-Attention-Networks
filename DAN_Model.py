import cv2
import glob
import json
import numpy as np
import rasterio
from keras.models import Input, Model
from keras.layers.core import Dropout
from keras.layers import AveragePooling2D, BatchNormalization, concatenate, Conv2D, Conv2DTranspose, Multiply
from keras.optimizers import Adam
from osgeo import gdal



def training_mask_generation(input_image_filename, input_geojson_filename):
    """ 
    This function is used to create a binary raster mask from polygons in a given geojson file, so as to label the pixels 
    in the image as either background or target.
    
    Inputs:
    - input_image_filename: File path of georeferenced image file to be used for model training
    - input_geojson_filename: File path of georeferenced geojson file which contains the polygons drawn over the targets
    
    Outputs:
    - mask: Numpy array representing the training mask, with values of 0 for background pixels, and value of 1 for target 
            pixels.
    
    """
    
    image = gdal.Open(input_image_filename)
    mask = np.zeros((image.RasterYSize, image.RasterXSize))
    
    ulx, xres, xskew, uly, yskew, yres = image.GetGeoTransform()                                   
    lrx = ulx + (image.RasterXSize * xres)                                                         
    lry = uly - (image.RasterYSize * abs(yres))

    polygons = json.load(open(input_geojson_filename))
    
    for polygon in range(len(polygons['features'])):
        coords = np.array(polygons['features'][polygon]['geometry']['coordinates'][0][0])                      
        xf = ((image.RasterXSize) ** 2 / (image.RasterXSize + 1)) / (lrx - ulx)
        yf = ((image.RasterYSize) ** 2 / (image.RasterYSize + 1)) / (lry - uly)
        coords[:, 1] = yf * (coords[:, 1] - uly)
        coords[:, 0] = xf * (coords[:, 0] - ulx)                                       
        position = np.round(coords).astype(np.int32)
        cv2.fillConvexPoly(mask, position, 1)
    
    return mask



def image_clip_to_segment_and_convert(image_array, mask_array, image_height_size, image_width_size, mode, percentage_overlap, 
                                      buffer):
    """ 
    This function is used to cut up images of any input size into segments of a fixed size, with empty clipped areas 
    padded with zeros to ensure that segments are of equal fixed sizes and contain valid data values. The function then 
    returns a 4 - dimensional array containing the entire image and its mask in the form of fixed size segments. 
    
    Inputs:
    - image_array: Numpy array representing the image to be used for model training (channels last format)
    - mask_array: Numpy array representing the binary raster mask to mark out background and target pixels
    - image_height_size: Height of image segments to be used for model training
    - image_width_size: Width of image segments to be used for model training
    - mode: Integer representing the status of image size
    - percentage_overlap: Percentage of overlap between image patches extracted by sliding window to be used for model 
                          training
    - buffer: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - image_segment_array: 4 - Dimensional numpy array containing the image patches extracted from input image array
    - mask_segment_array: 4 - Dimensional numpy array containing the mask patches extracted from input binary raster mask
    
    """
    
    y_size = ((image_array.shape[0] // image_height_size) + 1) * image_height_size
    x_size = ((image_array.shape[1] // image_width_size) + 1) * image_width_size
    
    if mode == 0:
        img_complete = np.zeros((y_size, image_array.shape[1], image_array.shape[2]))
        mask_complete = np.zeros((y_size, mask_array.shape[1], 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 1:
        img_complete = np.zeros((image_array.shape[0], x_size, image_array.shape[2]))
        mask_complete = np.zeros((image_array.shape[0], x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 2:
        img_complete = np.zeros((y_size, x_size, image_array.shape[2]))
        mask_complete = np.zeros((y_size, x_size, 1))
        img_complete[0 : image_array.shape[0], 0 : image_array.shape[1], 0 : image_array.shape[2]] = image_array
        mask_complete[0 : mask_array.shape[0], 0 : mask_array.shape[1], 0] = mask_array
    elif mode == 3:
        img_complete = image_array
        mask_complete = mask_array
        
    img_list = []
    mask_list = []
    
    
    for i in range(0, int(img_complete.shape[0] - (2 - buffer) * image_height_size), 
                   int((1 - percentage_overlap) * image_height_size)):
        for j in range(0, int(img_complete.shape[1] - (2 - buffer) * image_width_size), 
                       int((1 - percentage_overlap) * image_width_size)):
            M_90 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 90, 1.0)
            M_180 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 180, 1.0)
            M_270 = cv2.getRotationMatrix2D((image_width_size / 2, image_height_size / 2), 270, 1.0)
            img_original = img_complete[i : i + image_height_size, j : j + image_width_size, 0 : image_array.shape[2]]
            img_rotate_90 = cv2.warpAffine(img_original, M_90, (image_height_size, image_width_size))
            img_rotate_180 = cv2.warpAffine(img_original, M_180, (image_width_size, image_height_size))
            img_rotate_270 = cv2.warpAffine(img_original, M_270, (image_height_size, image_width_size))
            img_flip_hor = cv2.flip(img_original, 0)
            img_flip_vert = cv2.flip(img_original, 1)
            img_flip_both = cv2.flip(img_original, -1)
            img_list.extend([img_original, img_rotate_90, img_rotate_180, img_rotate_270, img_flip_hor, img_flip_vert, 
                             img_flip_both])
            mask_original = mask_complete[i : i + image_height_size, j : j + image_width_size, 0]
            mask_rotate_90 = cv2.warpAffine(mask_original, M_90, (image_height_size, image_width_size))
            mask_rotate_180 = cv2.warpAffine(mask_original, M_180, (image_width_size, image_height_size))
            mask_rotate_270 = cv2.warpAffine(mask_original, M_270, (image_height_size, image_width_size))
            mask_flip_hor = cv2.flip(mask_original, 0)
            mask_flip_vert = cv2.flip(mask_original, 1)
            mask_flip_both = cv2.flip(mask_original, -1)
            mask_list.extend([mask_original, mask_rotate_90, mask_rotate_180, mask_rotate_270, mask_flip_hor, mask_flip_vert, 
                              mask_flip_both])
    
    image_segment_array = np.zeros((len(img_list), image_height_size, image_width_size, image_array.shape[2]))
    mask_segment_array = np.zeros((len(mask_list), image_height_size, image_width_size, 1))
    
    for index in range(len(img_list)):
        image_segment_array[index] = img_list[index]
        mask_segment_array[index, :, :, 0] = mask_list[index]
        
    return image_segment_array, mask_segment_array



def training_data_generation(DATA_DIR, img_height_size, img_width_size, perc, buff):
    """ 
    This function is used to convert image files and their respective polygon training masks into numpy arrays, so as to 
    facilitate their use for model training.
    
    Inputs:
    - DATA_DIR: File path of folder containing the image files, and their respective polygons in a subfolder
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - perc: Percentage of overlap between image patches extracted by sliding window to be used for model training
    - buff: Percentage allowance for image patch to be populated by zeros for positions with no valid data values
    
    Outputs:
    - img_full_array: 4 - Dimensional numpy array containing image patches extracted from all image files for model training
    - mask_full_array: 4 - Dimensional numpy array containing binary raster mask patches extracted from all polygons for 
                       model training
    """
    
    if perc < 0 or perc > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for perc.')
        
    if buff < 0 or buff > 1:
        raise ValueError('Please input a number between 0 and 1 (inclusive) for buff.')
    
    img_files = glob.glob(DATA_DIR + '\\' + 'Train_*.tif')
    polygon_files = glob.glob(DATA_DIR + '\\Training Polygons' + '\\Train_*.geojson')
    
    img_array_list = []
    mask_array_list = []
    
    for file in range(len(img_files)):
        img = np.transpose(gdal.Open(img_files[file]).ReadAsArray(), [1, 2, 0])
        mask = training_mask_generation(img_files[file], polygon_files[file])
    
        if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 0, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 1, 
                                                                      percentage_overlap = perc, buffer = buff)
        elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 2, 
                                                                      percentage_overlap = perc, buffer = buff)
        else:
            img_array, mask_array = image_clip_to_segment_and_convert(img, mask, img_height_size, img_width_size, mode = 3, 
                                                                      percentage_overlap = perc, buffer = buff)
        
        img_array_list.append(img_array)
        mask_array_list.append(mask_array)
        
    img_full_array = np.concatenate(img_array_list, axis = 0)
    mask_full_array = np.concatenate(mask_array_list, axis = 0)
    
    return img_full_array, mask_full_array



def DAN_Model(img_height_size, img_width_size, n_bands, initial_conv_layers, growth_rate, dropout_rate, l_r,
              trans_down_1_size, trans_down_2_size, trans_down_3_size, trans_down_4_size, bottleneck_1_2_size, 
              bottleneck_3_4_size):
    """ 
    This function is used to generate the Dense Attention Network (DAN) architecture as described in the paper 'Building 
    Extraction in Very High Resolution Imagery by Dense - Attention Networks' by Yang H., Wu P., Yao X., Wu Y., Wang B., 
    Xu Y. (2018)
    
    Inputs:
    - img_height_size: Height of image patches to be used for model training
    - img_width_size: Width of image patches to be used for model training
    - n_bands: Number of channels contained in the image patches to be used for model training
    - initial_conv_layers: Number of convolutional layers to be used for the very first convolutional layer
    - growth_rate: Number of convolutional layers to be used for each layer in each dense block
    - dropout_rate: Dropout rate to be used during model training
    - l_r: Learning rate to be applied for the Adam optimizer
    - trans_down_1_size: Output number for feature maps for transition down level 1
    - trans_down_2_size: Output number for feature maps for transition down level 2
    - trans_down_3_size: Output number for feature maps for transition down level 3
    - trans_down_4_size: Output number for feature maps for transition down level 4
    - bottleneck_1_2_size: Output number for feature maps for bottleneck layers 1 and 2
    - bottleneck_3_4_size: Output number for feature maps for bottleneck layers 3 and 4
    
    Outputs:
    - dan_model: Dense Attention Network (DAN) model to be trained using input parameters and network architecture
        
    """
    
    block_1_size = initial_conv_layers + 2 * growth_rate
    block_2_size = trans_down_1_size + 2 * growth_rate
    block_3_size = trans_down_2_size + 3 * growth_rate
    block_4_size = trans_down_3_size + 3 * growth_rate
    block_5_size = trans_down_4_size + 3 * growth_rate

    img_input = Input(shape = (img_height_size, img_width_size, n_bands))
    batch_norm_initial = BatchNormalization()(img_input)
    conv_initial = Conv2D(initial_conv_layers, (7, 7), padding = 'same', activation = 'relu')(batch_norm_initial)
    
    batch_norm_1_1 = BatchNormalization()(conv_initial)
    layer_1_1 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_1_1)
    conv_1_layer_1_1 = concatenate([batch_norm_1_1, layer_1_1])
    batch_norm_1_2 = BatchNormalization()(conv_1_layer_1_1)
    layer_1_2 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_1_2)
    dense_block_1 = concatenate([batch_norm_1_1, layer_1_1, layer_1_2])
    
    batch_norm_down_1 = BatchNormalization()(dense_block_1)
    conv_down_1 = Conv2D(trans_down_1_size, (1, 1), padding = 'same', activation = 'relu')(batch_norm_down_1)
    conv_down_1 = Dropout(dropout_rate)(conv_down_1)
    trans_down_1 = AveragePooling2D(pool_size = (2, 2))(conv_down_1)
    
    batch_norm_2_1 = BatchNormalization()(trans_down_1)
    layer_2_1 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_2_1)
    conv_2_layer_2_1 = concatenate([batch_norm_2_1, layer_2_1])
    batch_norm_2_2 = BatchNormalization()(conv_2_layer_2_1)
    layer_2_2 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_2_2)
    dense_block_2 = concatenate([batch_norm_2_1, layer_2_1, layer_2_2])
    
    batch_norm_down_2 = BatchNormalization()(dense_block_2)
    conv_down_2 = Conv2D(trans_down_2_size, (1, 1), padding = 'same', activation = 'relu')(batch_norm_down_2)
    conv_down_2 = Dropout(dropout_rate)(conv_down_2)
    trans_down_2 = AveragePooling2D(pool_size = (2, 2))(conv_down_2)
    
    batch_norm_3_1 = BatchNormalization()(trans_down_2)
    layer_3_1 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_3_1)
    conv_3_layer_3_1 = concatenate([batch_norm_3_1, layer_3_1])
    batch_norm_3_2 = BatchNormalization()(conv_3_layer_3_1)
    layer_3_2 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_3_2)
    conv_3_layer_3_2 = concatenate([batch_norm_3_1, layer_3_1, layer_3_2])
    batch_norm_3_3 = BatchNormalization()(conv_3_layer_3_2)
    layer_3_3 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_3_3)
    dense_block_3 = concatenate([batch_norm_3_1, layer_3_1, layer_3_2, layer_3_3])
    
    batch_norm_down_3 = BatchNormalization()(dense_block_3)
    conv_down_3 = Conv2D(trans_down_3_size, (1, 1), padding = 'same', activation = 'relu')(batch_norm_down_3)
    conv_down_3 = Dropout(dropout_rate)(conv_down_3)
    trans_down_3 = AveragePooling2D(pool_size = (2, 2))(conv_down_3)
    
    batch_norm_4_1 = BatchNormalization()(trans_down_3)
    layer_4_1 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_4_1)
    conv_4_layer_4_1 = concatenate([batch_norm_4_1, layer_4_1])
    batch_norm_4_2 = BatchNormalization()(conv_4_layer_4_1)
    layer_4_2 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_4_2)
    conv_4_layer_4_2 = concatenate([batch_norm_4_1, layer_4_1, layer_4_2])
    batch_norm_4_3 = BatchNormalization()(conv_4_layer_4_2)
    layer_4_3 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_4_3)
    dense_block_4 = concatenate([batch_norm_4_1, layer_4_1, layer_4_2, layer_4_3])
    
    batch_norm_down_4 = BatchNormalization()(dense_block_4)
    conv_down_4 = Conv2D(trans_down_4_size, (1, 1), padding = 'same', activation = 'relu')(batch_norm_down_4)
    conv_down_4 = Dropout(dropout_rate)(conv_down_4)
    trans_down_4 = AveragePooling2D(pool_size = (2, 2))(conv_down_4)
    
    batch_norm_5_1 = BatchNormalization()(trans_down_4)
    layer_5_1 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_5_1)
    conv_5_layer_5_1 = concatenate([batch_norm_5_1, layer_5_1])
    batch_norm_5_2 = BatchNormalization()(conv_5_layer_5_1)
    layer_5_2 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_5_2)
    conv_5_layer_5_2 = concatenate([batch_norm_5_1, layer_5_1, layer_5_2])
    batch_norm_5_3 = BatchNormalization()(conv_5_layer_5_2)
    layer_5_3 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_5_3)
    dense_block_5 = concatenate([batch_norm_5_1, layer_5_1, layer_5_2, layer_5_3])
    
    deconv_block_5 = Conv2DTranspose(block_5_size, (2, 2), strides = (2, 2), padding = 'same', 
                                     activation = 'relu')(dense_block_5)
    sigmoid_block_5 = Conv2D(block_4_size, (1, 1), padding = 'same', activation = 'sigmoid')(deconv_block_5)
    weighted_block_4 = Multiply()([dense_block_4, sigmoid_block_5])
    spat_attn_fusion_1 = concatenate([deconv_block_5, weighted_block_4])
    
    batch_norm_6_1 = BatchNormalization()(spat_attn_fusion_1)
    layer_6_1 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_6_1)
    conv_6_layer_6_1 = concatenate([batch_norm_6_1, layer_6_1])
    batch_norm_6_2 = BatchNormalization()(conv_6_layer_6_1)
    layer_6_2 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_6_2)
    conv_6_layer_6_2 = concatenate([batch_norm_6_1, layer_6_1, layer_6_2])
    batch_norm_6_3 = BatchNormalization()(conv_6_layer_6_2)
    layer_6_3 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_6_3)
    dense_block_6 = concatenate([batch_norm_6_1, layer_6_1, layer_6_2, layer_6_3])
    bottleneck_1 = Conv2D(bottleneck_1_2_size, (1, 1), padding = 'same', activation = 'relu')(dense_block_6)
    bottleneck_1 = Dropout(dropout_rate)(bottleneck_1)
    
    deconv_bottleneck_1 = Conv2DTranspose(bottleneck_1_2_size, (2, 2), strides = (2, 2), padding = 'same', 
                                          activation = 'relu')(bottleneck_1)
    sigmoid_bottleneck_1 = Conv2D(block_3_size, (1, 1), padding = 'same', activation = 'sigmoid')(deconv_bottleneck_1)
    weighted_block_3 = Multiply()([dense_block_3, sigmoid_bottleneck_1])
    spat_attn_fusion_2 = concatenate([deconv_bottleneck_1, weighted_block_3])
    
    batch_norm_7_1 = BatchNormalization()(spat_attn_fusion_2)
    layer_7_1 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_7_1)
    conv_7_layer_7_1 = concatenate([batch_norm_7_1, layer_7_1])
    batch_norm_7_2 = BatchNormalization()(conv_7_layer_7_1)
    layer_7_2 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_7_2)
    conv_7_layer_7_2 = concatenate([batch_norm_7_1, layer_7_1, layer_7_2])
    batch_norm_7_3 = BatchNormalization()(conv_7_layer_7_2)
    layer_7_3 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_7_3)
    dense_block_7 = concatenate([batch_norm_7_1, layer_7_1, layer_7_2, layer_7_3])
    bottleneck_2 = Conv2D(bottleneck_1_2_size, (1, 1), padding = 'same', activation = 'relu')(dense_block_7)
    bottleneck_2 = Dropout(dropout_rate)(bottleneck_2)
    
    deconv_bottleneck_2 = Conv2DTranspose(bottleneck_1_2_size, (2, 2), strides = (2, 2), padding = 'same', 
                                          activation = 'relu')(bottleneck_2)
    sigmoid_bottleneck_2 = Conv2D(block_2_size, (1, 1), padding = 'same', activation = 'sigmoid')(deconv_bottleneck_2)
    weighted_block_2 = Multiply()([dense_block_2, sigmoid_bottleneck_2])
    spat_attn_fusion_3 = concatenate([deconv_bottleneck_2, weighted_block_2])
    
    batch_norm_8_1 = BatchNormalization()(spat_attn_fusion_3)
    layer_8_1 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_8_1)
    conv_8_layer_8_1 = concatenate([batch_norm_8_1, layer_8_1])
    batch_norm_8_2 = BatchNormalization()(conv_8_layer_8_1)
    layer_8_2 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_8_2)
    conv_8_layer_8_2 = concatenate([batch_norm_8_1, layer_8_1, layer_8_2])
    batch_norm_8_3 = BatchNormalization()(conv_8_layer_8_2)
    layer_8_3 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_8_3)
    dense_block_8 = concatenate([batch_norm_8_1, layer_8_1, layer_8_2, layer_8_3])
    bottleneck_3 = Conv2D(bottleneck_3_4_size, (1, 1), padding = 'same', activation = 'relu')(dense_block_8)
    bottleneck_3 = Dropout(dropout_rate)(bottleneck_3)
    
    deconv_bottleneck_3 = Conv2DTranspose(bottleneck_3_4_size, (2, 2), strides = (2, 2), padding = 'same', 
                                          activation = 'relu')(bottleneck_3)
    sigmoid_bottleneck_3 = Conv2D(block_1_size, (1, 1), padding = 'same', activation = 'sigmoid')(deconv_bottleneck_3)
    weighted_block_1 = Multiply()([dense_block_1, sigmoid_bottleneck_3])
    spat_attn_fusion_4 = concatenate([deconv_bottleneck_3, weighted_block_1])
    
    batch_norm_9_1 = BatchNormalization()(spat_attn_fusion_4)
    layer_9_1 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_9_1)
    conv_9_layer_9_1 = concatenate([batch_norm_9_1, layer_9_1])
    batch_norm_9_2 = BatchNormalization()(conv_9_layer_9_1)
    layer_9_2 = Conv2D(growth_rate, (3, 3), padding = 'same', activation = 'relu')(batch_norm_9_2)
    dense_block_9 = concatenate([batch_norm_9_1, layer_9_1, layer_9_2])
    bottleneck_4 = Conv2D(bottleneck_3_4_size, (1, 1), padding = 'same', activation = 'relu')(dense_block_9)
    bottleneck_4 = Dropout(dropout_rate)(bottleneck_4)
    
    pred_layer = Conv2D(1, (1, 1), padding = 'same', activation = 'sigmoid')(bottleneck_4)
    
    dan_model = Model(inputs = img_input, outputs = pred_layer)
    dan_model.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = l_r), metrics = ['binary_crossentropy'])
    
    return dan_model



def image_model_predict(input_image_filename, output_filename, img_height_size, img_width_size, fitted_model, write):
    """ 
    This function cuts up an image into segments of fixed size, and feeds each segment to the model for prediction. The 
    output mask is then allocated to its corresponding location in the image in order to obtain the complete mask for the 
    entire image without being constrained by image size. 
    
    Inputs:
    - input_image_filename: File path of image file for which prediction is to be conducted
    - output_filename: File path of output predicted binary raster mask file
    - img_height_size: Height of image patches to be used for model prediction
    - img_height_size: Width of image patches to be used for model prediction
    - fitted_model: Trained keras model which is to be used for prediction
    - write: Boolean indicating whether to write predicted binary raster mask to file
    
    Output:
    - mask_complete: Numpy array of predicted binary raster mask for input image
    
    """
    
    with rasterio.open(input_image_filename) as f:
        img = np.transpose(f.read(1))
        metadata = f.profile
    
    y_size = ((img.shape[0] // img_height_size) + 1) * img_height_size
    x_size = ((img.shape[1] // img_width_size) + 1) * img_width_size
    
    if (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size == 0):
        img_complete = np.zeros((y_size, img.shape[1], img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size == 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((img.shape[0], x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    elif (img.shape[0] % img_height_size != 0) and (img.shape[1] % img_width_size != 0):
        img_complete = np.zeros((y_size, x_size, img.shape[2]))
        img_complete[0 : img.shape[0], 0 : img.shape[1], 0 : img.shape[2]] = img
    else:
         img_complete = img
            
    mask = np.zeros((img_complete.shape[0], img_complete.shape[1], 1))
    img_holder = np.zeros((1, img_height_size, img_width_size, img.shape[2]))
    
    for i in range(0, img_complete.shape[0], img_height_size):
        for j in range(0, img_complete.shape[1], img_width_size):
            img_holder[0] = img_complete[i : i + img_height_size, j : j + img_width_size, 0 : img.shape[2]]
            preds = fitted_model.predict(img_holder)
            mask[i : i + img_height_size, j : j + img_width_size, 0] = preds[0, :, :, 0]
            
    mask_complete = np.transpose(mask[0 : img.shape[0], 0 : img.shape[1], 0], [2, 0, 1])
    
    if write:
        metadata['count'] = 1
        
        with rasterio.open(output_filename, 'w', **metadata) as dst:
            dst.write(mask_complete)
    
    return mask_complete