#coding: utf-8
"""U-net模型"""
from __future__ import print_function;
from keras.models import Model;
from keras.layers import Input, Dropout, Activation;
from keras.layers.convolutional import Conv2D, UpSampling2D;
from keras.layers.pooling import MaxPooling2D;
from keras.layers.merge import concatenate, add;
from resnet import ResnetBuilder;
from graph_utils import Graph_utils;
# from keras.utils.vis_utils import plot_model;

class UnetBuilder():
    
    @staticmethod
    def build(input_shape, filters, classes, class_mode="categorical"):
        
        assert classes >= 2;
        assert class_mode in ["binary", "categorical"];
        if classes > 2:
            assert class_mode == "categorical";
        
        inputs = Input(shape=input_shape);
        conv1 = Graph_utils._conv_relu_conv_relu(filters[0])(inputs);
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1);

        conv2 = Graph_utils._conv_relu_conv_relu(filters[1])(pool1);
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2);

        conv3 = Graph_utils._conv_relu_conv_relu(filters[2])(pool2);
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3);

        conv4 = Graph_utils._conv_relu_conv_relu(filters[3])(pool3);
        drop4 = Dropout(0.5)(conv4);
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4);

        conv5 = Graph_utils._conv_relu_conv_relu(filters[4])(pool4);
        drop5 = Dropout(0.5)(conv5);

        up6 = Conv2D(filters[3], kernel_size=(2, 2), activation="relu", 
                     padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(drop5));
        merge6 = concatenate([drop4,up6], axis = 3);
        conv6 = Graph_utils._conv_relu_conv_relu(filters[3])(merge6);
        
        up7 = Conv2D(filters[2], kernel_size=(2, 2), activation="relu", 
                     padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv6));
        merge7 = concatenate([conv3,up7], axis = 3);
        conv7 = Graph_utils._conv_relu_conv_relu(filters[2])(merge7);
        
        up8 = Conv2D(filters[1], kernel_size=(2, 2), activation="relu", 
                     padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv7));
        merge8 = concatenate([conv2,up8], axis = 3);
        conv8 = Graph_utils._conv_relu_conv_relu(filters[1])(merge8);
        
        up9 = Conv2D(filters[0], kernel_size=(2, 2), activation="relu", 
                     padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv8));
        merge9 = concatenate([conv1,up9], axis = 3);
        conv9 = Graph_utils._conv_relu_conv_relu(filters[0])(merge9);
        
        if class_mode == "binary":
            conv9 = Conv2D(classes, kernel_size=(3, 3), activation="relu", 
                           padding="same", kernel_initializer="he_normal")(conv9);
            conv10 = Conv2D(1, kernel_size=(1, 1), activation = "sigmoid", kernel_initializer="he_normal")(conv9);
        else:
            conv9 = Conv2D(classes, kernel_size=(3, 3), activation="relu", 
                           padding="same", kernel_initializer="he_normal")(conv9);
            conv9 = Conv2D(classes, kernel_size=(1, 1), padding="same", kernel_initializer="he_normal")(conv9);
            conv10 = Activation("softmax")(conv9);
        
        model = Model(inputs = inputs, outputs = conv10);
        return model;
        
    @staticmethod
    def build_resnet_unet(input_shape, classes, class_mode="categorical"):
        
        assert classes >= 2;
        assert class_mode in ["binary", "categorical"];
        if classes > 2:
            assert class_mode == "categorical";
        
        filters = [512, 256, 128, 64, 64]
        resnet_blocks, inputs = ResnetBuilder.build_resnet_18_for_other_model(input_shape, original_resnet=False);
        h = [resnet_blocks[4], resnet_blocks[3], resnet_blocks[2], resnet_blocks[1], resnet_blocks[0]];
        
        for i in range(5):
            h[i] = Conv2D(filters=filters[i], kernel_size=(3, 3), padding="same")(h[i]);
            
        up6 = Conv2D(filters[1], kernel_size=(3, 3), activation="relu", 
                     padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(h[0]));
        merge6 = add([h[1],up6]);
        conv6 = Graph_utils._conv_relu_conv_relu(filters[1])(merge6);
        
        up7 = Conv2D(filters[2], kernel_size=(3, 3), activation="relu", 
                     padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv6));
        merge7 = add([h[2],up7]);
        conv7 = Graph_utils._conv_relu_conv_relu(filters[2])(merge7);
 
        up8 = Conv2D(filters[3], kernel_size=(3, 3), activation="relu", 
                     padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv7));
        merge8 = add([h[3], up8]);
        conv8 = Graph_utils._conv_relu_conv_relu(filters[3])(merge8);
 
        up9 = Conv2D(filters[4], kernel_size=(3, 3), activation="relu", 
                     padding="same", kernel_initializer="he_normal")(UpSampling2D(size = (2,2))(conv8));
        merge9 = add([h[4], up9]);
        conv9 = Graph_utils._conv_relu_conv_relu(filters[4])(merge9);
        
        if class_mode == "binary":
            conv9 = Conv2D(classes, kernel_size=(3, 3), activation="relu", 
                           padding="same", kernel_initializer="he_normal")(conv9);
            conv9 = Conv2D(1, kernel_size=(1, 1), padding="same", kernel_initializer="he_normal")(conv9);
            conv10 = Activation("sigmoid")(conv9);
        else:
            conv9 = Conv2D(classes, kernel_size=(3, 3), activation="relu", 
                           padding="same", kernel_initializer="he_normal")(conv9);
            conv9 = Conv2D(classes, kernel_size=(1, 1), padding="same", kernel_initializer="he_normal")(conv9);
            conv10 = Activation("softmax")(conv9);
        
        
        model = Model(inputs = inputs, outputs = conv10);
        return model;   
        
# model = UnetBuilder.build((256, 256, 3), [64, 128, 256, 512, 1024], 3);
# model.summary()
# plot_model(model, to_file="unet.png", show_shapes=True);
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

