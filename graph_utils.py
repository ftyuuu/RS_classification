#coding: utf-8
"""graph工具"""
from __future__ import print_function;
from __future__ import division;
from keras.engine import Layer, InputSpec;
from keras.layers import Conv2D, Input, Activation, Dense, Flatten, UpSampling2D;
from keras.layers.pooling import AveragePooling2D, MaxPooling2D;
from keras.layers.normalization import BatchNormalization;
from keras.regularizers import l2;
from keras.layers.merge import add;
from keras import backend as K;
import tensorflow as tf;

epsilon = 1.e-9;

class Graph_utils():
    
    @staticmethod
    def _conv_relu_conv_relu(filters, padding="same", kernel_size=(3, 3), 
                             kernel_initializer="he_normal"):
        
        def f(input):
            net = Conv2D(filters, kernel_size=kernel_size, padding=padding, 
                         kernel_initializer=kernel_initializer)(input);
            net = Activation("relu")(net);
            net = Conv2D(filters, kernel_size=kernel_size, padding=padding, 
                         kernel_initializer=kernel_initializer)(net);
            net = Activation("relu")(net);
            return net;
        return f;
    
    @staticmethod
    def _bn_relu():
        def f(input):
            norm = BatchNormalization(axis=3)(input);
            return Activation("relu")(norm);
        return f;
    
    @staticmethod
    def _conv_bn_relu(filters, padding="same", kernel_size=(3, 3), 
                      kernel_initializer="he_normal"):
        def f(input):
            net = Conv2D(filters, kernel_size=kernel_size, padding=padding, 
                         kernel_initializer=kernel_initializer)(input);
            net = Graph_utils._bn_relu()(net);
            return net;
        return f;
    
    @staticmethod
    def _res_conv_bn_relu_conv_bn_relu(filters, padding="same", kernel_size=(3, 3), 
                                       kernel_initializer="he_normal"):
        def f(input):
            net = Graph_utils._conv_bn_relu(filters, kernel_size=kernel_size, padding=padding, 
                                            kernel_initializer=kernel_initializer)(input);
            net = Graph_utils._conv_bn_relu(filters, kernel_size=kernel_size, padding=padding, 
                                            kernel_initializer=kernel_initializer)(net);
            net = add([input, net]);
            return net;
        return f;
    
    
class BatchNorm(BatchNormalization):

    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=True);

    
    