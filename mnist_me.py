# -*- coding: utf-8 -*-
import numpy as np
import time,sys,getopt
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

class GAN(object):
    def __init__(self):
        self.generator = self._generator_layer()
        self.discriminator = self._discriminator_layer()
        self.DM = self._discriminator_model(self.discriminator)
        self.AM = self._AM_model(self.generator,self.discriminator)
        self.x_data = None
        self.batch_size = 32
        
    def _generator_layer(self):
        model = Sequential()
        model.add(BatchNormalization(input_shape=(100,)))
        model.add(Dense(7*7*64*4,activation='relu'))
        model.add(Reshape((7,7,64*4)))
        model.add(Dropout(0.4))   
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(64*2), 5, padding='same',activation='relu'))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(64), 5, padding='same',activation='relu'))            
        model.add(Conv2DTranspose(int(32), 5, padding='same',activation='relu'))  
        model.add(Conv2DTranspose(1, 5, padding='same',activation='sigmoid'))    
        return model
    def _discriminator_layer(self):
        model = Sequential()
        model.add(Conv2D(64, 5, strides=2, input_shape=(28,28,1),padding='same',activation='relu'))
        model.add(Dropout(0.3))
        model.add(Conv2D(128, 5, strides=2 ,padding='same',activation='relu'))
        model.add(Dropout(0.3))
        model.add(Conv2D(64*4, 5, strides=2 ,padding='same',activation='relu'))
        model.add(Dropout(0.3))        
        model.add(Conv2D(64*8, 5, strides=1 ,padding='same',activation='relu'))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(1,activation='sigmoid'))
        return model
    def _AM_model(self,generator,discriminator):
        optimizer = RMSprop(lr=0.001, decay=1e-6)
        AM = Sequential()
        AM.add(generator)
        AM.add(discriminator)
        AM.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        return AM
        
    def _discriminator_model(self,model):
        DM = Sequential()
        optimizer = RMSprop(lr=0.002, decay=1e-6)
        DM.add(model)
        DM.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])
        return DM
    def train(self,data,train_steps=2000,savename = 'test',plot = True):
        for train_step in range(train_steps):
            noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, 100])
            x_batch = data[np.random.randint(0,data.shape[0],[self.batch_size]),:,:,:]
            x_fake = self.generator.predict(noise)
            x = np.concatenate((x_batch, x_fake))
            y = np.ones([2*self.batch_size, 1])
            y[self.batch_size:, :] = 0
            d_loss = self.DM.train_on_batch(x, y)
            
            y = np.ones([self.batch_size, 1])
            #noise = np.random.uniform(-1.0, 1.0, size=[self.batch_size, 100])
            a_loss = self.AM.train_on_batch(noise, y)
            if (train_step+1) % 10 == 0:
                if plot:
                    self.plotFake(4)
                self.save_weights(savename)
                print('{}:  D:[loss:{}  acc:{}]   A:[loss:{}  acc:{}]'.format(train_step+1,d_loss[0],d_loss[1],a_loss[0],a_loss[1]))

    def plotFake(self,num=16):
        noise = np.random.uniform(-1.0, 1.0, size=[num, 100])
        fakeimg = self.generator.predict(noise)
        H = int(np.floor(np.sqrt(num)))
        L = int(np.ceil(np.sqrt(num)))
        for i in range(H*L):
            plt.subplot(H,L,i+1)
            plt.imshow(fakeimg[i,:,:,:].reshape([28,28]), cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def save_weights(self,name):
        self.generator.save_weights(name+'.generator')
        self.discriminator.save_weights(name+'.discriminator')
        
    def load_weights(self,name):
        self.generator.load_weights(name+'.generator')
        self.discriminator.load_weights(name+'.discriminator')
    
    def __str__(self):
        return str(self.generator.summary())+'\n\n'+str(self.discriminator.summary())

def plotImgs(imgs):
    num = len(imgs)
    H = int(np.floor(np.sqrt(num)))
    L = int(np.ceil(np.sqrt(num)))
    for i in range(H*L):
        plt.subplot(H,L,i+1)
        plt.imshow(imgs[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()


    
if __name__=="__main__":      
    optiondata = '''
    
    -R                 run training.
    
    -i [nums]          train iter (2000).
    -r [filename]      read weight filename.
    -w [filename]      wright weight filename. default:`read weight filename` or 'test'
    
    -h                 help
    '''
    opts, args = getopt.getopt(sys.argv[1:], "hRr:i:w:")
    opts = dict(opts)
    if '-h' in opts:
        print(optiondata)
        sys.exit()
    if '-R' in opts:
        x_train = input_data.read_data_sets("mnist",one_hot=True).train.images
        x_train = x_train.reshape([len(x_train),28,28,1])
        
        gan = GAN()
        loadweightname = opts.get('-r') 
        if loadweightname:
            try:
                gan.load_weights(loadweightname)
                print('[Message] LOAD weight name `{}`'.format(loadweightname))
            except Exception:
                print('[Warning] `{}`data does not exists.'.format(loadweightname))
                pass
        saveweightname = opts.get('-w') if opts.get('-w') else 'temp' if not loadweightname else loadweightname
        trainiter = int(opts.get('-i')) if opts.get('-i') else 2000
        gan.train(x_train, train_steps=trainiter, savename=saveweightname,plot = False)
    else:# spyder debuging
        x_train = input_data.read_data_sets("mnist",one_hot=True).train.images
        x_train = x_train.reshape([len(x_train),28,28,1])
        gan = GAN()








