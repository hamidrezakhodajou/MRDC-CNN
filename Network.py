import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation, add
from keras.models import Model, model_from_json
from keras import backend as K 

def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), activation='relu', name=None):
    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, kernel_initializer='glorot_uniform')(x)
    if(activation == None):
        return x
    x = Activation(activation, name=name)(x)
    return x

def MRF(W, inp):
    shortcut = conv2d_bn(inp, 2*W, 1, 1, activation=None, padding='same')
    conv3x3 = conv2d_bn(inp, W, 3, 3,activation='relu', padding='same')
    conv5x5 = conv2d_bn(conv3x3, W, 3, 3, activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5], axis=3)
    out = add([shortcut, out])
    out = Activation('relu')(out)
    return out


def DenseSkipConnection(filters, length, inp, kernelSize):
    if length<=0:
        return inp
    elif length>0:
        out = conv2d_bn(inp, filters, kernelSize, kernelSize, activation='relu', padding='same')
        inp = concatenate([out, inp], axis=3)
    
        for i in range(length-1):
            out = conv2d_bn(inp, filters, kernelSize, kernelSize, activation='relu', padding='same')
            inp = concatenate([out, inp], axis=3)
        return inp

               

def MRDC_CNN(height, width, n_channels, filters=16, kernelSize=3, base_Layer=4):

    inputs = Input((height, width, n_channels))

    mresblock1 = MRF(filters, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(mresblock1)
    mresblock1 = DenseSkipConnection(filters, base_Layer, mresblock1, kernelSize)

    mresblock2 = MRF(filters*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(mresblock2)
    mresblock2 = DenseSkipConnection(filters*2, base_Layer-1, mresblock2, kernelSize)

    mresblock3 = MRF(filters*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(mresblock3)
    mresblock3 = DenseSkipConnection(filters*4, base_Layer-2, mresblock3, kernelSize)

    mresblock4 = MRF(filters*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(mresblock4)
    mresblock4 = DenseSkipConnection(filters*8, base_Layer-3, mresblock4, kernelSize)

    mresblock5 = MRF(filters*16, pool4)

    up6 = concatenate([Conv2DTranspose(
        filters*8, (2,2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(mresblock5), mresblock4], axis=3)
    mresblock6 = MRF(filters*8, up6)

    up7 = concatenate([Conv2DTranspose(
        filters*4, (2,2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(mresblock6), mresblock3], axis=3)
    mresblock7 = MRF(filters*4, up7)

    up8 = concatenate([Conv2DTranspose(
        filters*2, (2,2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(mresblock7), mresblock2], axis=3)
    mresblock8 = MRF(filters*2, up8)

    up9 = concatenate([Conv2DTranspose(
        filters, (2,2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(mresblock8), mresblock1], axis=3)
    mresblock9 = MRF(filters, up9)
    
    
    conv10 = conv2d_bn(mresblock9, 1,1,1, activation='linear',name= '512') #512
    conv11 = conv2d_bn(mresblock8, 1,1,1, activation='linear', name= '256') #256
    conv12 = conv2d_bn(mresblock7, 1,1,1, activation='linear', name= '128') #128
    conv13 = conv2d_bn(mresblock6, 1,1,1, activation='linear', name= '64') #64

    model = Model(inputs=[inputs], outputs=[conv10, conv11, conv12, conv13])
    return model





