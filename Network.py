import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Activation, add
from keras.models import Model


def conv2d(x, filters, row_kernel, col_kernel, padding='same', strides=(1, 1), activation='relu', name=None):
    x = Conv2D(filters, (row_kernel, col_kernel), strides=strides, padding=padding, use_bias=False, kernel_initializer='glorot_uniform')(x)
    if(activation == None):
        return x
    x = Activation(activation, name=name)(x)
    return x

def MRF(filters, inp):
    shortcut = conv2d(inp, 2*filters, 1, 1, activation=None, padding='same')
    conv3x3 = conv2d(inp, filters, 3, 3,activation='relu', padding='same')
    conv5x5 = conv2d(conv3x3, filters, 3, 3, activation='relu', padding='same')

    out = concatenate([conv3x3, conv5x5], axis=3)
    out = add([shortcut, out])
    out = Activation('relu')(out)
    return out


def DenseSkipConnection(filters, length, inp, kernelSize):
    if length<=0:
        return inp
    elif length>0:
        out = conv2d(inp, filters, kernelSize, kernelSize, activation='relu', padding='same')
        inp = concatenate([out, inp], axis=3)
    
        for i in range(length-1):
            out = conv2d(inp, filters, kernelSize, kernelSize, activation='relu', padding='same')
            inp = concatenate([out, inp], axis=3)
        return inp

               

def MRDC_CNN(height, width, n_channels, base_filter=16, kernelSize=3, base_Layer=4):

    inputs = Input((height, width, n_channels))

    MRFblock1 = MRF(base_filter, inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(MRFblock1)
    MRFblock1 = DenseSkipConnection(base_filter, base_Layer, MRFblock1, kernelSize)

    MRFblock2 = MRF(base_filter*2, pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(MRFblock2)
    MRFblock2 = DenseSkipConnection(base_filter*2, base_Layer-1, MRFblock2, kernelSize)

    MRFblock3 = MRF(base_filter*4, pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(MRFblock3)
    MRFblock3 = DenseSkipConnection(base_filter*4, base_Layer-2, MRFblock3, kernelSize)

    MRFblock4 = MRF(base_filter*8, pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(MRFblock4)
    MRFblock4 = DenseSkipConnection(base_filter*8, base_Layer-3, MRFblock4, kernelSize)

    MRFblock5 = MRF(base_filter*16, pool4)

    up6 = concatenate([Conv2DTranspose(
        base_filter*8, (2,2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(MRFblock5), MRFblock4], axis=3)
    MRFblock6 = MRF(base_filter*8, up6)

    up7 = concatenate([Conv2DTranspose(
        base_filter*4, (2,2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(MRFblock6), MRFblock3], axis=3)
    MRFblock7 = MRF(base_filter*4, up7)

    up8 = concatenate([Conv2DTranspose(
        base_filter*2, (2,2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(MRFblock7), MRFblock2], axis=3)
    MRFblock8 = MRF(base_filter*2, up8)

    up9 = concatenate([Conv2DTranspose(
        base_filter, (2,2), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(MRFblock8), MRFblock1], axis=3)
    MRFblock9 = MRF(base_filter, up9)
    
    
    out_512 = conv2d(MRFblock9, 1,1,1, activation='linear',name= '512') #512
    out_256 = conv2d(MRFblock8, 1,1,1, activation='linear', name= '256') #256
    out_128 = conv2d(MRFblock7, 1,1,1, activation='linear', name= '128') #128
    out_64 = conv2d(MRFblock6, 1,1,1, activation='linear', name= '64') #64

    model = Model(inputs=[inputs], outputs=[out_512, out_256, out_128, out_64])
    return model
