#coding: utf-8
"""Resnet模型"""
from __future__ import print_function;
from keras.models import Model;
from keras.layers.convolutional import Conv2D;
from keras.layers import Input, Dense, Flatten;
from keras.layers.pooling import AveragePooling2D, MaxPooling2D;
from keras import backend as K;
from keras.regularizers import l2;
from keras.layers.merge import add;
from graph_utils import Graph_utils;

 
def _conv_bn_relu(**conv_params):
    filters = conv_params["filters"];
    kernel_size = conv_params["kernel_size"];
    strides = conv_params.setdefault("strides", (1, 1));
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal");
    padding = conv_params.setdefault("padding", "same");
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4));
 
    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input);
        return Graph_utils._bn_relu()(conv);
 
    return f;
 
def _bn_relu_conv(**conv_params):
    filters = conv_params["filters"];
    kernel_size = conv_params["kernel_size"];
    strides = conv_params.setdefault("strides", (1, 1));
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal");
    padding = conv_params.setdefault("padding", "same");
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4));
 
    def f(input):
        activation = Graph_utils._bn_relu()(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation);
 
    return f;
 
def _shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[1] / float(residual_shape[1])));
    stride_height = int(round(input_shape[2] / float(residual_shape[2])));
    equal_channels = input_shape[3] == residual_shape[3];
 
    shortcut = input;
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input);
 
    return add([shortcut, residual]);
 
def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1);
            if i == 0 and not is_first_layer:
                init_strides = (2, 2);
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input);
        return input;
 
    return f;
 
def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):
 
        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input);
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input);
 
        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1);
        return _shortcut(input, residual);
 
    return f;
 
def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    def f(input):
        if is_first_block_of_first_layer:
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input);
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input);
 
        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1);
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3);
        return _shortcut(input, residual);
    return f;

class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions, 
              logits_and_block_endpoints=False, original_resnet=False, 
              input_tensor=None):
        if input_shape is None and input_tensor is None:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        assert K.image_dim_ordering() == 'tf';
        
        if input_tensor is None:
            input = Input(shape=input_shape, name="input_image");
        else:
            input = input_tensor;
            
        resnet_blocks_endpoints = [];
        
        if original_resnet:
            conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2), name="resnet_initial_conv1")(input);
            conv1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same", name="resnet_initial_pool1")(conv1);
        else:
            conv1 = _conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(1, 1), name="resnet_initial_conv1")(input);
            conv1 = _conv_bn_relu(filters=64, kernel_size=(3, 3), strides=(1, 1), name="resnet_initial_conv2")(conv1);
            if logits_and_block_endpoints:
                resnet_blocks_endpoints.append(conv1);
            conv1 = MaxPooling2D(pool_size=(2, 2), padding="same", name="resnet_initial_pool1")(conv1);

        block = conv1;
        filters = 64;
        
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block);
            filters *= 2;
            
            if logits_and_block_endpoints:
                """Save each residual block."""
                resnet_blocks_endpoints.append(block);
        
        if logits_and_block_endpoints:
            return resnet_blocks_endpoints, input;
           
        block = Graph_utils._bn_relu()(block);

        block_shape = K.int_shape(block);
        pool2 = AveragePooling2D(pool_size=(block_shape[1], block_shape[2]),
                                 strides=(1, 1))(block);
        flatten1 = Flatten()(pool2);
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1);

        model = Model(inputs=input, outputs=dense);
        return model;
    
    @staticmethod
    def build_resnet_18(input_shape, num_outputs, original_resnet=False):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2], original_resnet=original_resnet)

    @staticmethod
    def build_resnet_34(input_shape, num_outputs, original_resnet=False):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3], original_resnet=original_resnet)

    @staticmethod
    def build_resnet_50(input_shape, num_outputs, original_resnet=False):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3], original_resnet=original_resnet)

    @staticmethod
    def build_resnet_18_for_other_model(input_shape, original_resnet=False, input_tensor=None):
        resnet_blocks, input = ResnetBuilder.build(input_shape, None, basic_block,\
                                                   [2, 2, 2, 2], logits_and_block_endpoints=True,\
                                                   original_resnet=original_resnet, input_tensor=input_tensor);
        return resnet_blocks, input;

# model = ResnetBuilder.build_resnet_18((256, 256, 3), 2, original_resnet=True)
# model.summary();
# from keras.utils.vis_utils import plot_model;
# plot_model(model, to_file="resnet.png", show_shapes=True);













 
