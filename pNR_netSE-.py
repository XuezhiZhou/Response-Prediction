# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:46:20 2022

@author: pc
"""
import scipy.misc
import numpy as np
import pandas as pd
import random
from sklearn.utils import shuffle
import cv2
from PIL import Image
import os
import csv
import keras
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Flatten, Reshape, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, Activation, Permute, multiply
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
import time
# from tqdm import tqdm, trange
from tensorflow.keras.preprocessing import image
from keras.layers.core import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
import h5py
import pickle
from collections import Counter
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import transform
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import imgaug as ia
from imgaug import augmenters as iaa

os.chdir('E:\\project\\pNR_rectalcancer\\attention_radiomics\\SE-\\')
# ia.seed(1)

def normalization(data, a, b):
    range_num = np.max(data) - np.min(data)
    new = (data-np.min(data)) / range_num * (b-a) + a
    return new

seq = iaa.Sequential([
    iaa.Fliplr(0.2), 
    iaa.Flipud(0.2), 
    iaa.Crop(percent=(0, 0.05)), # random crops
    iaa.Sometimes(0.2,
        iaa.GaussianBlur(sigma=(0, 0.05))
    ),
    iaa.LinearContrast((0.8, 1.2), per_channel=0.2),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.02*1), per_channel=0.2),
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True)

# train_generator = get_data_batch(train_sample_path, train_mask_path, batch_size=5, img_rows=img_rows, img_cols=img_cols, img_channels=4, augm="True")
# im = next(train_generator)
# plt.imshow(im[0][0,:,:,0])

def SEBlock(se_ratio = 1, data_format = 'channels_last', ki = "he_normal"):
    def f(input_x):
        channel_axis = -1 if data_format == 'channels_last' else 1
        input_channels = np.shape(input_x)[channel_axis]
        reduced_channels = input_channels // se_ratio
        #Squeeze operation
        x = GlobalAveragePooling2D()(input_x)
        x = Reshape(1,1,input_channels)(x) if data_format == 'channels_first' else x
        x = Dense(reduced_channels, kernel_initializer= ki, activation='relu')(x)
        #Excitation operation
        x = Dense(input_channels, kernel_initializer=ki, activation='sigmoid')(x)
        x = Permute(dims=(3,1,2))(x) if data_format == 'channels_first' else x
        x = multiply([input_x, x])
        return x
    return f

def pNRnet(input_img):
    x = BatchNormalization()(input_img)
    # x = SEBlock(se_ratio=0.5)(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv1')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)

    x = BatchNormalization()(x)
    # x = SEBlock(se_ratio=0.5)(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)

    x = BatchNormalization()(x)
    # x = SEBlock(se_ratio=0.5)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)

    x = Flatten()(x)
    x = Dense(64, activation='relu', name='dense1')(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_img, outputs=y)
    return model

def get_sample_path(sample_list, sample_dir_path, label_list):
    sample_path = []
    for i in range(len(sample_list)):
        file_path_list = []
        file_path_ADC = os.path.join(sample_dir_path, "pre-"+sample_list[i]+"-ADC.nii")
        file_path_T1C = os.path.join(sample_dir_path, "pre-"+sample_list[i]+"-T1C.nii")
        file_path_T2WI = os.path.join(sample_dir_path, "pre-"+sample_list[i]+"-T2WI.nii")
        file_path_list.append([file_path_ADC, file_path_T1C, file_path_T2WI, label_list[i]])
        sample_path.extend(file_path_list)
    return sample_path

def get_mask_path(sample_list, sample_dir_path, label_list):
    sample_path = []
    for i in range(len(sample_list)):
        file_path_list = []
        file_path_ADC = os.path.join(sample_dir_path, "pre-"+sample_list[i]+"-mask.nii")
        file_path_T1C = os.path.join(sample_dir_path, "pre-"+sample_list[i]+"-mask.nii")
        file_path_T2WI = os.path.join(sample_dir_path, "pre-"+sample_list[i]+"-mask.nii")
        file_path_list.append([file_path_ADC, file_path_T1C, file_path_T2WI, label_list[i]])
        sample_path.extend(file_path_list)
    return sample_path

def get_data_batch(shufflefile_image, shufflefile_mask, batch_size, img_rows, img_cols, img_channels, augm = "False"):
    while 1:
        for i in range(0, len(shufflefile_image), batch_size):
            x = []
            y = []
            if (i + batch_size) > len(shufflefile_image):
                                
                for j in range(len(shufflefile_image[i:len(shufflefile_image)])):

                    I=np.empty(shape=[img_rows, img_cols, img_channels])
                    
                    file_image = shufflefile_image[i:len(shufflefile_image)][j]
                    file_mask = shufflefile_mask[i:len(shufflefile_mask)][j]
                    
                    image_0 = sitk.ReadImage(file_image[0])
                    image_0_data = sitk.GetArrayFromImage(image_0)
                    image_0_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_0_data,axis=2),80,80)
                    image_1 = sitk.ReadImage(file_image[1])
                    image_1_data = sitk.GetArrayFromImage(image_1)
                    image_1_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_1_data,axis=2),80,80)
                    image_2 = sitk.ReadImage(file_image[2])
                    image_2_data = sitk.GetArrayFromImage(image_2)
                    image_2_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_2_data,axis=2),80,80)

                    mask_0 = sitk.ReadImage(file_mask[0])
                    mask_0_data = sitk.GetArrayFromImage(mask_0)
                    mask_0_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_0_data,axis=2),80,80)
                    mask_1 = sitk.ReadImage(file_mask[1])
                    mask_1_data = sitk.GetArrayFromImage(mask_1)
                    mask_1_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_1_data,axis=2),80,80)
                    mask_2 = sitk.ReadImage(file_mask[2])
                    mask_2_data = sitk.GetArrayFromImage(mask_2)
                    mask_2_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_2_data,axis=2),80,80)
                   
                    # scaler = StandardScaler()
                    I[:,:,0] = normalization(transform.resize((image_0_data * mask_0_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)
                    I[:,:,1] = normalization(transform.resize((image_1_data * mask_1_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)
                    I[:,:,2] = normalization(transform.resize((image_2_data * mask_2_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)

                    I = np.array(I)
                    x.append(I)
                    y.append(file_image[3])

            else:                   
                for j in range(len(shufflefile_image[i:i + batch_size])):

                    I=np.empty(shape=[img_rows, img_cols, img_channels])
                    
                    file_image = shufflefile_image[i:i + batch_size][j]
                    file_mask = shufflefile_mask[i:i + batch_size][j]
                    
                    image_0 = sitk.ReadImage(file_image[0])
                    image_0_data = sitk.GetArrayFromImage(image_0)
                    image_0_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_0_data,axis=2),80,80)
                    image_1 = sitk.ReadImage(file_image[1])
                    image_1_data = sitk.GetArrayFromImage(image_1)
                    image_1_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_1_data,axis=2),80,80)
                    image_2 = sitk.ReadImage(file_image[2])
                    image_2_data = sitk.GetArrayFromImage(image_2)
                    image_2_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_2_data,axis=2),80,80)
                    
                    mask_0 = sitk.ReadImage(file_mask[0])
                    mask_0_data = sitk.GetArrayFromImage(mask_0)
                    mask_0_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_0_data,axis=2),80,80)
                    mask_1 = sitk.ReadImage(file_mask[1])
                    mask_1_data = sitk.GetArrayFromImage(mask_1)
                    mask_1_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_1_data,axis=2),80,80)
                    mask_2 = sitk.ReadImage(file_mask[2])
                    mask_2_data = sitk.GetArrayFromImage(mask_2)
                    mask_2_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_2_data,axis=2),80,80)

                    # scaler = StandardScaler()
                    I[:,:,0] = normalization(transform.resize((image_0_data * mask_0_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)
                    I[:,:,1] = normalization(transform.resize((image_1_data * mask_1_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)
                    I[:,:,2] = normalization(transform.resize((image_2_data * mask_2_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)
                    
                    I = np.array(I)
                    x.append(I)
                    y.append(file_image[3])
                    
            x = np.array(x).reshape(len(x), img_rows, img_cols, img_channels)
            x = x.astype('float32')
            if (augm == "True"):
                x = seq.augment_images(x)
            y = np.array(y)
            yield x, y

def auc_curve(y, prob, set):
    fpr, tpr, threshold = roc_curve(y, prob)  
    roc_auc = auc(fpr, tpr) 
    #plt.figure()
    lw = 2
    #plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUC: %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(set)
    plt.legend(loc="lower right")
    #plt.show()
    return roc_auc

def label_proba_plot(label, prob):
    plt.plot(label,prob,"r.",markersize=2,zorder=1,label=u"真实值")
    plt.xlim([-0.25, 1.25])
    plt.ylim([0.0, 1.0])
    plt.xlabel('label')
    plt.ylabel('predict_proba')

############grad-CAM
def grad_cam(model, image_input, category_index, layer_name):
    """
    Args:
       model: model
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """
    heatmap_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output[:, category_index]])
    with tf.GradientTape() as gtape:
        convolution_output, prob = heatmap_model(image_input)
        grads = gtape.gradient(prob, convolution_output)[0]
        castConvOutputs = tf.cast(convolution_output > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
    
    output, grads_val = convolution_output[0], guidedGrads[0]
    weights = tf.reduce_mean(grads_val, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, output), axis=-1)
    
    heatmap = cv2.resize(cam.numpy(), (image_input.shape[1], image_input.shape[2]))
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    return np.uint8(cam), heatmap

def plot_gradCAM(model, sample_path, mask_path, layer_name="pool3", img_rows=64, img_cols=64, channels=3, vmin=0, vmax=0.04):
    for i in range(len(sample_path)): 
        I=np.empty(shape=[img_rows, img_cols, channels])
        file_image = sample_path[i]
        file_mask = mask_path[i]
        image_0 = sitk.ReadImage(file_image[0])
        image_0_data = sitk.GetArrayFromImage(image_0)
        image_0_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_0_data,axis=2),80,80)
        image_1 = sitk.ReadImage(file_image[1])
        image_1_data = sitk.GetArrayFromImage(image_1)
        image_1_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_1_data,axis=2),80,80)
        image_2 = sitk.ReadImage(file_image[2])
        image_2_data = sitk.GetArrayFromImage(image_2)
        image_2_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_2_data,axis=2),80,80)

        mask_0 = sitk.ReadImage(file_mask[0])
        mask_0_data = sitk.GetArrayFromImage(mask_0)
        mask_0_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_0_data,axis=2),80,80)
        mask_1 = sitk.ReadImage(file_mask[1])
        mask_1_data = sitk.GetArrayFromImage(mask_1)
        mask_1_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_1_data,axis=2),80,80)
        mask_2 = sitk.ReadImage(file_mask[2])
        mask_2_data = sitk.GetArrayFromImage(mask_2)
        mask_2_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_2_data,axis=2),80,80)
        I[:,:,0] = normalization(transform.resize((image_0_data * mask_0_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)
        I[:,:,1] = normalization(transform.resize((image_1_data * mask_1_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)
        I[:,:,2] = normalization(transform.resize((image_2_data * mask_2_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)
        
        I = np.array(I)
        I = np.array(I).reshape(1, img_rows, img_cols, channels)
        prob = model.predict(I)
        cam, heatmap = grad_cam(model, I, 0, layer_name)
        plt.figure()
        plt.suptitle(str(str(sample_path[i][0].split("\\")[-1].split("-")[1]))+", label="+str(sample_path[i][3])+", prob="+str(prob[0][0]), fontsize=20, y=1.05)
        ax = plt.subplot(2,3,1)
        ax.set_title("ADC")
        plt.axis("off")
        plt.imshow(I[0,:,:,0],cmap=plt.cm.gray)
        ax = plt.subplot(2,3,2)
        ax.set_title("T1C")
        plt.axis("off")
        plt.imshow(I[0,:,:,1],cmap=plt.cm.gray)
        ax = plt.subplot(2,3,3)
        ax.set_title("T2WI")
        plt.axis("off")
        plt.imshow(I[0,:,:,2],cmap=plt.cm.gray)
        ax = plt.subplot(2,3,4)
        ax.set_title("grad-CAM")
        plt.axis("off")
        plt.imshow(heatmap,cmap=plt.cm.jet, vmin=vmin, vmax=vmax)
        ax = plt.subplot(2,3,5)
        plt.axis("off")
        plt.colorbar(shrink=1) 

def plot_SEblock(model, sample_path, mask_path, layer_name="multiply", img_rows=64, img_cols=64, channels=3, vmin=0, vmax=0.04):
    SE_outset=[]
    for i in range(len(sample_path)): #len(sample_path)       
        I=np.empty(shape=[img_rows, img_cols, channels])
        file_image = sample_path[i]
        file_mask = mask_path[i]
        image_0 = sitk.ReadImage(file_image[0])
        image_0_data = sitk.GetArrayFromImage(image_0)
        image_0_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_0_data,axis=2),80,80)
        image_1 = sitk.ReadImage(file_image[1])
        image_1_data = sitk.GetArrayFromImage(image_1)
        image_1_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_1_data,axis=2),80,80)
        image_2 = sitk.ReadImage(file_image[2])
        image_2_data = sitk.GetArrayFromImage(image_2)
        image_2_data = tf.image.resize_with_crop_or_pad(np.expand_dims(image_2_data,axis=2),80,80)

        mask_0 = sitk.ReadImage(file_mask[0])
        mask_0_data = sitk.GetArrayFromImage(mask_0)
        mask_0_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_0_data,axis=2),80,80)
        mask_1 = sitk.ReadImage(file_mask[1])
        mask_1_data = sitk.GetArrayFromImage(mask_1)
        mask_1_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_1_data,axis=2),80,80)
        mask_2 = sitk.ReadImage(file_mask[2])
        mask_2_data = sitk.GetArrayFromImage(mask_2)
        mask_2_data = tf.image.resize_with_crop_or_pad(np.expand_dims(mask_2_data,axis=2),80,80)

        I[:,:,0] = normalization(transform.resize((image_0_data * mask_0_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)
        I[:,:,1] = normalization(transform.resize((image_1_data * mask_1_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)
        I[:,:,2] = normalization(transform.resize((image_2_data * mask_2_data)[:,:,0], (img_rows, img_cols), order=3), 0, 1)

        I = np.array(I)
        I = np.array(I).reshape(1, img_rows, img_cols, channels)
        layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        SE_out = layer_model.predict(I)
        if (len(SE_out.shape)>2):
            plt.figure()
            plt.suptitle(str(str(sample_path[i][0].split("\\")[-1].split("-")[1]))+", label="+str(sample_path[i][3]), fontsize=20, y=1.05)
            ax = plt.subplot(3,3,1)
            ax.set_title("ADC")
            plt.axis("off")
            plt.imshow(I[0,:,:,0],cmap=plt.cm.gray,vmin=vmin,vmax=vmax)
            ax = plt.subplot(3,3,2)
            ax.set_title("T1C")
            plt.axis("off")
            plt.imshow(I[0,:,:,1],cmap=plt.cm.gray,vmin=vmin,vmax=vmax)
            ax = plt.subplot(3,3,3)
            ax.set_title("T2WI")
            plt.axis("off")
            plt.imshow(I[0,:,:,2],cmap=plt.cm.gray,vmin=vmin,vmax=vmax)
            ax = plt.subplot(3,3,4)
            ax.set_title("SE-ADC")
            plt.axis("off")
            plt.imshow(SE_out[0,:,:,0],cmap=plt.cm.gray,vmin=vmin,vmax=vmax)
            ax = plt.subplot(3,3,5)
            ax.set_title("SE-T1C")
            plt.axis("off")
            plt.imshow(SE_out[0,:,:,1],cmap=plt.cm.gray,vmin=vmin,vmax=vmax)
            ax = plt.subplot(3,3,6)
            ax.set_title("SE-T2WI")
            plt.axis("off")
            plt.imshow(SE_out[0,:,:,2],cmap=plt.cm.gray,vmin=vmin,vmax=vmax)
            ax = plt.subplot(3,3,7)
            plt.axis("off")
            plt.colorbar(shrink=1) 
        else:
            print(str(str(sample_path[i][0].split("\\")[-1].split("-")[1]))+", label="+str(sample_path[i][3]))
            # print(SE_out.shape)
            print(SE_out[0,:])
            SE_outset.append(SE_out[0,:]) 

################## &&&&&&&&&&&&&&&&&&&&&&&&&&&& #################
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
data_dir_image_train = "E:\\project\\pNR_rectalcancer\\attention_radiomics\\data_cut_zhongshan\\"
data_dir_mask_train = "E:\\project\\pNR_rectalcancer\\attention_radiomics\\mask_cut_zhongshan\\"

data_dir_image_test = "E:\\project\\pNR_rectalcancer\\attention_radiomics\\data_cut_zhejiang\\"
data_dir_mask_test = "E:\\project\\pNR_rectalcancer\\attention_radiomics\\mask_cut_zhejiang\\"

csv_data = pd.read_excel("E:\\project\\pNR_rectalcancer\\attention_radiomics\\clinical_zhongshan.xlsx", sheet_name="Sheet1",header=0,skiprows=None,index_col=None,names=None)
sample_name_train = list(csv_data['sample'])
pNR_label_train = list(csv_data['TRG_group'])

csv_data = pd.read_excel("E:\\project\\pNR_rectalcancer\\attention_radiomics\\clinical_zhejiang.xlsx", sheet_name="Sheet2",header=0,skiprows=None,index_col=None,names=None)
sample_name_test = list(csv_data['sample'])
pNR_label_test = list(csv_data['TRG_group'])

train_sample_path = get_sample_path(sample_name_train, data_dir_image_train, pNR_label_train)
train_mask_path = get_mask_path(sample_name_train, data_dir_mask_train, pNR_label_train)

test_sample_path = get_sample_path(sample_name_test, data_dir_image_test, pNR_label_test)
test_mask_path = get_mask_path(sample_name_test, data_dir_mask_test, pNR_label_test)

img_rows = 64
img_cols = 64
batch_size = 20
tuneModel = pNRnet(Input(shape=(img_rows, img_cols, 3)))
tuneModel.compile(loss='binary_crossentropy',
                  optimizer=SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['AUC'])

t0 = time.time()
train_epoch_num = 1000
list_train_auc = np.empty(shape=train_epoch_num)
list_validation_auc = np.empty(shape=train_epoch_num)
list_test_auc = np.empty(shape=train_epoch_num)

rand_num = 193
train_sample_path_1, validation_sample_path_1, pNR_label_train_1, pNR_label_validation_1 = train_test_split(train_sample_path,pNR_label_train,test_size=0.3,random_state=rand_num)
train_mask_path_1, validation_mask_path_1, pNR_label_train_1, pNR_label_validation_1 = train_test_split(train_mask_path,pNR_label_train,test_size=0.3,random_state=rand_num) 

for epoch in range(0, train_epoch_num):
    print('epoch:%d' % epoch)
    
    fitted_generator = get_data_batch(shuffle(train_sample_path_1, random_state=epoch),
    shuffle(train_mask_path_1, random_state=epoch),
    batch_size=batch_size, img_rows=img_rows, img_cols=img_cols, img_channels=3, augm="True")
    
    tuneModel.fit(fitted_generator, steps_per_epoch=np.ceil(len(train_sample_path_1) / batch_size), epochs=1, verbose=1)
    
    train_generator = get_data_batch(train_sample_path_1, train_mask_path_1, batch_size=batch_size, img_rows=img_rows, img_cols=img_cols, img_channels=3, augm="False")
    pred_train = tuneModel.predict(train_generator, steps=np.ceil(len(train_sample_path_1) / batch_size), workers=0, use_multiprocessing=True)
    train_predicted_label = np.round(pred_train)
    train_accuracy = accuracy_score(pNR_label_train_1, train_predicted_label)
    train_auc = roc_auc_score(pNR_label_train_1, pred_train)
    
    validation_generator = get_data_batch(validation_sample_path_1, validation_mask_path_1, batch_size=batch_size, img_rows=img_rows, img_cols=img_cols, img_channels=3, augm="False")
    pred_validation = tuneModel.predict(validation_generator, steps=np.ceil(len(validation_sample_path_1) / batch_size), workers=0, use_multiprocessing=True)
    validation_predicted_label = np.round(pred_validation)
    validation_accuracy = accuracy_score(pNR_label_validation_1, validation_predicted_label)
    validation_auc = roc_auc_score(pNR_label_validation_1, pred_validation)
    
    print('train-Accuracy: %.5f, train-AUC: %.5f' % (train_accuracy, train_auc))
    print('validation-Accuracy: %.5f, validation-AUC: %.5f' % (validation_accuracy, validation_auc))
    print('time: %.5f s' % (time.time() - t0))
    print('\n')
    list_train_auc[epoch] = train_auc
    list_validation_auc[epoch] = validation_auc
    
    tuneModel.save('pNR-weights-' + str('%03d' % (epoch)) + str('_%.5f' % (train_auc)) + str('_%.5f' % (validation_auc)) + '.hdf5')

plt.plot(list_train_auc)
plt.plot(list_validation_auc)
plt.title('model AUC')
plt.ylabel('AUC')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

with h5py.File(os.getcwd()+'\\epoch_AUC.h5','w') as f:
    f['list_train_auc'] = list_train_auc
    f['list_validation_auc'] = list_validation_auc

model = load_model('pNR-weights-084_0.85823_0.83537.hdf5')
model.summary()
train_generator = get_data_batch(train_sample_path_1, train_mask_path_1, batch_size=batch_size, img_rows=img_rows, img_cols=img_cols, img_channels=3, augm="False")
pred_train = model.predict(train_generator, steps=np.ceil(len(train_sample_path_1) / batch_size), workers=0, use_multiprocessing=True)
   
validation_generator = get_data_batch(validation_sample_path_1, validation_mask_path_1, batch_size=batch_size, img_rows=img_rows, img_cols=img_cols, img_channels=3, augm="False")
pred_validation = model.predict(validation_generator, steps=np.ceil(len(validation_sample_path_1) / batch_size), workers=0, use_multiprocessing=True)

test_generator = get_data_batch(test_sample_path, test_mask_path, batch_size=batch_size, img_rows=img_rows, img_cols=img_cols, img_channels=3, augm="False")
pred_test = model.predict(test_generator, steps=np.ceil(len(test_sample_path) / batch_size), workers=0, use_multiprocessing=True)
  
plt.figure(figsize=(9, 6))
plt.subplot(2, 3, 1, frameon = True) 
train_auc = auc_curve(pNR_label_train_1,pred_train,'train cohort')
plt.subplot(2, 3, 2, frameon = True) 
validation_auc = auc_curve(pNR_label_validation_1,pred_validation,'validation cohort')
plt.subplot(2, 3, 3, frameon = True) 
test_auc = auc_curve(pNR_label_test,pred_test,'test cohort')
plt.subplot(2, 3, 4, frameon = True) 
label_proba_plot(pNR_label_train_1,pred_train)
plt.subplot(2, 3, 5, frameon = True) 
label_proba_plot(pNR_label_validation_1,pred_validation)
plt.subplot(2, 3, 6, frameon = True) 
label_proba_plot(pNR_label_test,pred_test)
plt.tight_layout()

plot_gradCAM(model, test_sample_path[0:30], test_mask_path[0:30], layer_name="multiply",
             img_rows=img_rows, img_cols=img_cols, channels=3, vmin=0, vmax=0.002)

plot_gradCAM(model, test_sample_path[0:30], test_mask_path[0:30], layer_name="pool3",
             img_rows=img_rows, img_cols=img_cols, channels=3, vmin=0, vmax=0.04)
