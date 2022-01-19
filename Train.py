VIEW = 120          # 30, 60, 120 
base_filter = 16    # Base Filters
kernelSize = 3      # Kernel Size
base_Layer = 4      # Base Layers

################################################################################################################################
import tensorflow as tf
from skimage.measure import block_reduce
from tensorflow.keras.optimizers import Adam

import numpy as np 
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

################################################################################################################################
SIZE = 512

train_X = np.load('Data/train_X_512_{}.npy'.format(VIEW),mmap_mode="r")
print('The size of train data:',train_X.shape)

train_Y512 = np.load('Data/train_Y_512.npy',mmap_mode="r")
train_Y256= np.reshape(block_reduce(train_Y512, (1, 2 , 2,1), np.mean), [len(train_X), 256,256,1])
train_Y128= np.reshape(block_reduce(train_Y256, (1, 2 , 2,1), np.mean), [len(train_X), 128,128,1])
train_Y64= np.reshape(block_reduce(train_Y128, (1, 2 , 2,1), np.mean), [len(train_X), 64,64,1])

print('Max_512: {:.1f}, Min_512: {:.1f}'.format(np.max(np.max(train_Y512)), np.min(np.min(train_Y512))))
print('Max_256: {:.1f}, Min_256: {:.1f}'.format(np.max(np.max(train_Y256)), np.min(np.min(train_Y256))))
print('Max_128: {:.1f}, Min_128: {:.1f}'.format(np.max(np.max(train_Y128)), np.min(np.min(train_Y128))))
print('Max_64 : {:.1f}, Min_64 : {:.1f}'.format(np.max(np.max(train_Y64)),  np.min(np.min(train_Y64))))

################################################################################################################################
from Network import MRDC_CNN

model = MRDC_CNN(height=512, width=512, n_channels=1,base_filter=base_filter, kernelSize=kernelSize, base_Layer=base_Layer)
model.summary()

################################################################################################################################
def SSIM_Acc (y_true, y_pred):
    max_y = 3071.0
    min_y = -1024.0
    data_range = max_y - min_y
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val= data_range))

def PSNR_Acc(y_true, y_pred):
    max_y = 3071.0
    min_y = -1024.0
    data_range = max_y - min_y
    return tf.image.psnr(y_true, y_pred, max_val=data_range)

def loss_512(y_true, y_pred):
    max_y = 3071.0
    min_y = -1024.0
    data_range = max_y - min_y
    
    loss1 = tf.reduce_mean(tf.math.subtract(tf.constant(1.0), tf.image.ssim(y_true, y_pred, max_val= data_range)))
    loss2 = loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=[1])/(data_range*data_range))
    loss = loss1 + 0.01*loss2
    return loss

def loss_256(y_true, y_pred):
    max_y = 3071.0
    min_y = -1024.0
    data_range = max_y - min_y
    
    loss1 = tf.reduce_mean(tf.math.subtract(tf.constant(1.0), tf.image.ssim(y_true, y_pred, max_val= data_range)))
    loss2 = loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=[1])/(data_range*data_range))
    loss = loss1 + 0.01*loss2
    return loss


def loss_128(y_true, y_pred):
    max_y = 3060.8
    min_y = -1024.0
    data_range = max_y - min_y
    
    loss1 = tf.reduce_mean(tf.math.subtract(tf.constant(1.0), tf.image.ssim(y_true, y_pred, max_val= data_range)))
    loss2 = loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=[1])/(data_range*data_range))
    loss = loss1 + 0.01*loss2
    return loss


def loss_64(y_true, y_pred):
    max_y = 1578.1
    min_y = -1024.0
    data_range = max_y - min_y
    
    loss1 = tf.reduce_mean(tf.math.subtract(tf.constant(1.0), tf.image.ssim(y_true, y_pred, max_val= data_range)))
    loss2 = loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_true - y_pred), axis=[1])/(data_range*data_range))
    loss = loss1 + 0.01*loss2
    return loss


# compile model
opt = Adam(lr=0.001)
model.compile(loss=[loss_512, loss_256, loss_128, loss_64], 
              loss_weights=[np.exp(0), np.exp(-1),np.exp(-2), np.exp(-3)], 
              optimizer=opt, metrics=[SSIM_Acc])

################################################################################################################################

print('Start Training')
epochs = 200
batch_size = 16
            
for epoch in range(epochs):
    print()
    print('Epoch : {}'.format(epoch+1))
    
    train_Y512 = np.cast['float32'](train_Y512)
    train_Y256 = np.cast['float32'](train_Y256)
    train_Y128 = np.cast['float32'](train_Y128)
    train_Y64 = np.cast['float32'](train_Y64)

    history = model.fit(train_X, [train_Y512, train_Y256, train_Y128, train_Y64], batch_size= batch_size, epochs=1, verbose=1) 
    
model.save('models/MRDCCNN_{}_view{}_.h5'.format(VIEW, SIZE, VIEW)) 
print('The network is saved.')    
print('End')
