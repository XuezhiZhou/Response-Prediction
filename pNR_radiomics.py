# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 22:58:24 2022

@author: pc
"""

import scipy.misc
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
from PIL import Image
import os
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
import keras
from keras.models import Model, load_model, Sequential
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet_v2 import MobileNetV2
# from keras.utils import multi_gpu_model
from keras import layers, regularizers
from keras.layers import Lambda, Input, Conv1D, Conv2D, MaxPooling2D, UpSampling2D, Dense, Dropout, Flatten, Reshape, BatchNormalization, AveragePooling2D, GlobalAveragePooling2D, Activation, Reshape, multiply
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, CSVLogger
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm, trange
from keras.engine.topology import Layer, InputSpec
import h5py
import pickle
from collections import Counter
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from keras import backend as K
import tensorflow as tf

def get_data_batch(patient_use, TRG_use, patient_name_set, patient_radiomics_set, batch_size, img_rows, img_cols, img_channels):
    while 1:
        for i in range(0, len(patient_use), batch_size):
            x = []
            y = []
            if (i + batch_size) > len(patient_use):
                name_sequence = patient_use[i:len(patient_use)]
            else:
                name_sequence = patient_use[i:(i+batch_size)]
            for sample in name_sequence:
                I=np.empty(shape=[img_rows, img_cols, img_channels])
                I[:,:,0]=patient_radiomics_set[patient_name_set.index('pre-'+sample+'-ADC.nii'),:]
                I=np.array(I)
                x.append(I)
                y.append(TRG_use[patient_use.index(sample)])

            x = np.array(x).reshape(len(x), img_rows, img_cols, img_channels)
            x = x.astype('float32')
            y = np.asarray(y)
            yield x, y
            
def get_data_nnRegression(patient_use, patient_name_set, patient_radiomics_set):
    x = []
    for sample in patient_use:
        I=patient_radiomics_set[patient_name_set.index('pre-'+sample+'-ADC.nii'),:]
        I=np.array(I)
        x.append(I)
    return np.asarray(x)
        
def Modality_attention(se_ratio = 1, num_modality = 3 ,ki = "he_normal"):
 
    def f(input_x):
        input_atom = np.shape(input_x)[-2]
        #attention operation
        x = Reshape((num_modality,int(input_atom/num_modality),1))(input_x)
        x = Conv2D(32,(num_modality,1),kernel_initializer= ki, padding = 'valid',activation='relu',name='conv1')(x)
        x = Conv2D(32,(1,1),kernel_initializer= ki, padding = 'valid',activation='relu',name='conv2')(x)
        x = Conv2D(1,(1,1),kernel_initializer= ki, padding = 'valid',activation='relu',name='conv3')(x)
        x = Flatten()(x)
        # Excitation operation
        x = Dense(num_modality, kernel_initializer=ki, activation='sigmoid',name='dense1')(x)
        x = Reshape((num_modality,1))(x)
        y = Reshape((int(input_atom/num_modality),num_modality))(input_x)
        z = K.batch_dot(y, x)
        return z
    return f

def pNRnet(input_feature):
    x = Modality_attention(se_ratio=1,num_modality=3)(input_feature)
    x = Flatten()(x)
    y = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_feature, outputs=y)
    return model

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

path_result = 'E:/project/pNR_rectalcancer/attention_radiomics/' 
os.getcwd()
os.chdir(path_result)

csv_data = pd.read_csv('E:/project/pNR_rectalcancer/attention_radiomics/py_2D_zhongshan.csv')
train_sample = list(csv_data['sample'])[0::4]
csv_data_radiomics = np.concatenate((csv_data.iloc[:,csv_data.columns.str.contains('original')],
                                      csv_data.iloc[:,csv_data.columns.str.contains('log')]),
                                    axis=1)
# csv_data_radiomics = csv_data.iloc[:,csv_data.columns.str.contains('original')]
s1 = np.array(csv_data_radiomics[0::4])
s2 = np.array(csv_data_radiomics[1::4])
s3 = np.array(csv_data_radiomics[3::4])
radiomics_train=np.concatenate((s1,s2,s3),axis=1)
scaler = MinMaxScaler()
scaler.fit(radiomics_train)
radiomics_train_scale = scaler.transform(radiomics_train)

csv_data = pd.read_csv('E:/project/pNR_rectalcancer/attention_radiomics/py_2D_zhejiang.csv')
test_sample = list(csv_data['sample'])[0::4]
csv_data_radiomics = np.concatenate((csv_data.iloc[:,csv_data.columns.str.contains('original')],
                                      csv_data.iloc[:,csv_data.columns.str.contains('log')]),
                                    axis=1)
# csv_data_radiomics = csv_data.iloc[:,csv_data.columns.str.contains('original')]
s1 = np.array(csv_data_radiomics[0::4])
s2 = np.array(csv_data_radiomics[1::4])
s3 = np.array(csv_data_radiomics[3::4])
radiomics_test=np.concatenate((s1,s2,s3),axis=1)
radiomics_test_scale = scaler.transform(radiomics_test)

clinical_data = pd.read_excel('E:/project/pNR_rectalcancer/attention_radiomics/clinical_zhongshan.xlsx',sheet_name='Sheet1')
sample_name_train = list(clinical_data['sample'])
TRG_group_train = list(clinical_data['TRG_group'])

clinical_data = pd.read_excel('E:/project/pNR_rectalcancer/attention_radiomics/clinical_zhejiang.xlsx',sheet_name='Sheet2')
sample_name_test = list(clinical_data['sample'])
TRG_group_test = list(clinical_data['TRG_group'])

###t-sne
# radiomics_all = np.concatenate((radiomics_train_scale,radiomics_test_scale),axis=0)
# radiomics_label =['SAHSYU' for index in range(np.shape(radiomics_train)[0])] + ['ZCH' for index in range(np.shape(radiomics_test)[0])]
# from sklearn import manifold
# import seaborn as sns 
# def visual(feature):
#     ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
#     x_ts = ts.fit_transform(feature)
#     print(x_ts.shape)  # [num, 2]
#     x_min, x_max = x_ts.min(0), x_ts.max(0)
#     x_final = (x_ts - x_min) / (x_max - x_min)
#     return x_final
# def plotlabels(S_lowDWeights, True_labels, name):
#     S_data = pd.DataFrame({'1st_Component': S_lowDWeights[:, 0], '2nd_Component': S_lowDWeights[:, 1], 'label': True_labels})
#     print(S_data)
#     print(S_data.shape)  # [num, 3]
#     sns.scatterplot(data=S_data, hue='label', x='1st_Component', y='2nd_Component') 
#     # plt.xticks([])
#     # plt.yticks([])
#     plt.title(name, fontsize=20, fontweight='normal', pad=20)
# fig = plt.figure(figsize=(10, 10))
# plotlabels(visual(radiomics_all), radiomics_label, 'dataset')
# plt.show(fig)

#################################lasso

rand_num = 193
train_sample_path_1, validation_sample_path_1, train_label_list_1, validation_label_list_1 = train_test_split(sample_name_train,
                                                                                            TRG_group_train,
                                                                                            test_size=0.3,
                                                                                            random_state=rand_num) 

radiomics_train_scale_=radiomics_train_scale
radiomics_test_scale_=radiomics_test_scale

feature_train_path_1 =get_data_nnRegression(train_sample_path_1,train_sample,radiomics_train_scale_)
feature_validation_path_1 =get_data_nnRegression(validation_sample_path_1,train_sample,radiomics_train_scale_)
feature_test_path =get_data_nnRegression(sample_name_test,test_sample,radiomics_test_scale_)

lr = LogisticRegressionCV(multi_class="ovr",fit_intercept=True,Cs=np.linspace(0.001,1,1000),refit=True,cv=5,penalty="l2",solver="liblinear",tol=0.001,random_state=20)
re = lr.fit(feature_train_path_1,train_label_list_1)

pred_train = re.predict_proba(feature_train_path_1)[:,1]
pred_validation = re.predict_proba(feature_validation_path_1)[:,1]
pred_test = re.predict_proba(feature_test_path)[:,1]

plt.figure(figsize=(9, 6))
plt.subplot(2, 3, 1, frameon = True) 
train_auc = auc_curve(train_label_list_1,pred_train,'train cohort')
plt.subplot(2, 3, 2, frameon = True) 
validation_auc = auc_curve(validation_label_list_1,pred_validation,'validation cohort')
plt.subplot(2, 3, 3, frameon = True) 
test_auc = auc_curve(TRG_group_test,pred_test,'test cohort')
plt.subplot(2, 3, 4, frameon = True) 
label_proba_plot(train_label_list_1,pred_train)
plt.subplot(2, 3, 5, frameon = True) 
label_proba_plot(validation_label_list_1,pred_validation)
plt.subplot(2, 3, 6, frameon = True) 
label_proba_plot(TRG_group_test,pred_test)
plt.tight_layout() 

#################################attention net
feature_num = s1.shape[1]
batch_size = 20
tuneModel = pNRnet(Input(shape=(1,feature_num*3,1)))
tuneModel.compile(loss='binary_crossentropy',
                  optimizer=SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True),
                  metrics=['AUC'])

t0 = time.time()
train_epoch_num = 1500
list_train_auc = np.empty(shape=train_epoch_num)
list_validation_auc = np.empty(shape=train_epoch_num)
list_test_auc = np.empty(shape=train_epoch_num)

for epoch in tqdm(range(0, train_epoch_num)):
    print('epoch:%d' % epoch)
    fitted_generator = get_data_batch(train_sample_path_1,
                                      train_label_list_1,
                                      train_sample,
                                      radiomics_train_scale,batch_size,1,feature_num*3,1)
    tuneModel.fit(fitted_generator, steps_per_epoch=np.ceil(len(train_label_list_1) / batch_size), epochs=1, verbose=1)
    # tuneModel.save('weights-' + str('%03d' % (epoch)) + '.hdf5')
    # tuneModel.summary()
    train_generator = get_data_batch(train_sample_path_1,train_label_list_1,train_sample,radiomics_train_scale,batch_size,1,feature_num*3,1)
    pred_train = tuneModel.predict(train_generator, steps=np.ceil(len(train_label_list_1) / batch_size), workers=0, use_multiprocessing=True)
    train_predicted_label = np.round(pred_train)
    train_accuracy = accuracy_score(train_label_list_1, train_predicted_label)
    train_auc = roc_auc_score(train_label_list_1, pred_train)
    
    validation_generator = get_data_batch(validation_sample_path_1,validation_label_list_1,train_sample,radiomics_train_scale,batch_size,1,feature_num*3,1)
    pred_validation = tuneModel.predict(validation_generator, steps=np.ceil(len(validation_label_list_1) / batch_size), workers=0, use_multiprocessing=True)
    validation_predicted_label = np.round(pred_validation)
    validation_accuracy = accuracy_score(validation_label_list_1, validation_predicted_label)
    validation_auc = roc_auc_score(validation_label_list_1, pred_validation)
    
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

model = load_model('pNR-weights-603_0.83595_0.81212.hdf5',custom_objects={'f': Modality_attention()})
model.summary()
train_generator = get_data_batch(train_sample_path_1,train_label_list_1,train_sample,radiomics_train_scale,batch_size,1,feature_num*3,1)
pred_train = model.predict(train_generator, steps=np.ceil(len(train_label_list_1) / batch_size), workers=0, use_multiprocessing=True)
validation_generator = get_data_batch(validation_sample_path_1,validation_label_list_1,train_sample,radiomics_train_scale,batch_size,1,feature_num*3,1)
pred_validation = model.predict(validation_generator, steps=np.ceil(len(validation_label_list_1) / batch_size), workers=0, use_multiprocessing=True)
test_generator = get_data_batch(sample_name_test,TRG_group_test,test_sample,radiomics_test_scale,batch_size,1,feature_num*3,1)
pred_test = model.predict(test_generator, steps=np.ceil(len(TRG_group_test) / batch_size), workers=0, use_multiprocessing=True) 
   
plt.figure(figsize=(9, 6))
plt.subplot(2, 3, 1, frameon = True)
train_auc = auc_curve(train_label_list_1,pred_train,'train cohort')
plt.subplot(2, 3, 2, frameon = True)
validation_auc = auc_curve(validation_label_list_1,pred_validation,'validation cohort')
plt.subplot(2, 3, 3, frameon = True)
test_auc = auc_curve(TRG_group_test,pred_test,'test cohort')
plt.subplot(2, 3, 4, frameon = True)
label_proba_plot(train_label_list_1,pred_train)
plt.subplot(2, 3, 5, frameon = True)
label_proba_plot(validation_label_list_1,pred_validation)
plt.subplot(2, 3, 6, frameon = True)
label_proba_plot(TRG_group_test,pred_test)
plt.tight_layout()