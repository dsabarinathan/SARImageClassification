# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 13:05:39 2022

@author: SABARI
"""

from efficientnet import EfficientNetB4
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Concatenate ,Conv2D, BatchNormalization,LeakyReLU
from keras.optimizers import Adam

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
#        x = LeakyReLU(alpha=0.1)(x)
        x = Activation('elu')(x)
    return x

def denseBlock(blockInput, num_filters=16, batch_activate = False,filer_size=(3,3)):
    count=3
    li = [blockInput]
    pas=convolution_block(blockInput, num_filters,size=filer_size,strides=(1,1))
    for i in range(2 , count+1):
        li.append(pas)
        out =  Concatenate(axis = 3)(li) # conctenated out put
        pas=convolution_block(out, num_filters,size=filer_size,strides=(1,1))
    # feature extractor from the dense net
    li.append(pas)
    out = Concatenate(axis =3)(li)
    feat=convolution_block(out, num_filters,size=filer_size,strides=(1,1))
    return feat

def modifiedEfficientNet(input_shape=(64,64,3),learningRate=0.0001,dropout_rate=0.1):

    backbone = EfficientNetB4(weights=None,
                            include_top=False,
                            input_shape=input_shape)
    
    
    inputs = backbone.input
    start_neurons = 16

    conv4 = backbone.layers[342].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same")(pool4)
    convm = denseBlock(convm,start_neurons * 32)
    convm = denseBlock(convm,start_neurons * 32)

    GA = GlobalAveragePooling2D()(convm)
    x1= BatchNormalization()(GA)
    x1= Dropout(0.2)(x1)
#    x2= BatchNormalization()(GA)
    x2= Dropout(0.2)(x1)
    output1= Dense(10,activation='softmax',name='normal')(x2)   
    model1 = Model(inputs, [output1])
    adam0 = Adam(lr=learningRate)

    model1.compile(optimizer=adam0, loss='sparse_categorical_crossentropy',metrics= ['acc'])

    return model1
