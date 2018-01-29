#coding: utf-8
'''Super pixel分类'''
from __future__ import print_function;
import numpy as np;
import math;
import argparse;
import tensorflow as tf;
import os;
import zipfile;

tiff_path = "/media/fty/Windows/linux_data/PAI_Demo/data/";
label_path = "/media/fty/Windows/linux_data/PAI_Demo/label/";
evaluated_path = "/media/fty/Windows/linux_data/PAI_Demo/evaluation/";


"""Shared parameters"""
train_ids = np.arange(2) + 1;
test_ids = [6];

input_shape = (64, 64, 3);
n_segments = 10;
batch_size = 8;
epochs = 1000;
load_weights_path = None;
loss_fun = "categorical_crossentropy"
loss_fun = "focal_loss_1d"

"""Resnet parameters"""
model_name = 'resnet';
model_alias_name = "resnet_super_pixel"
original_resnet = False;

"""PAI parameters"""
PAI = False;

if __name__ == '__main__':
    
    '''For PAI'''
    if PAI:
        print('For PAI');
        parser = argparse.ArgumentParser();
        parser.add_argument('--buckets', type=str, default='', help='input data path');
        parser.add_argument('--checkpointDir', type=str, default='', help='output model path');
        FLAGS, _ = parser.parse_known_args();
         
        print('Copy code and data to PAI');
        zip_name = 'remote_sensing.zip';
        tf.gfile.Copy(os.path.join(FLAGS.buckets, zip_name), 
                      os.path.join('./', zip_name), 
                      overwrite=True);
        f = zipfile.ZipFile(zip_name, 'r');
        for file in f.namelist():
            print(file);
            f.extract(file, './');
    else:
        FLAGS = None;
    
    from utils import Vaihingen_class;
    from utils import Super_pixel_seg_dataset;
    from model import ModelBuilder;
    plot = Vaihingen_class;
    active_positive_class = [];
    active_positive_class.append(Vaihingen_class.Building);
    active_positive_class.append(Vaihingen_class.Tree);
    active_positive_class.append(Vaihingen_class.Car);
    active_positive_class.append(Vaihingen_class.Low_vegetation);
    classes = len(active_positive_class) + 1;
    
        
    super_pixel_dataset_training = Super_pixel_seg_dataset(tiff_path, 
                                                           label_path,
                                                           plot,
                                                           active_positive_class);
    
    """You also can set parameters for different model."""
    model = ModelBuilder(PAI_FLAGS=FLAGS,
                         input_shape=input_shape, 
                         classes=classes, 
                         model_name=model_name,
                         model_alias_name=model_alias_name,
                         load_weights=load_weights_path,
                         original_resnet=original_resnet);
    
    super_pixel_dataset_training.prepare_superpixel_dataset(is_train=True,
                                                            load_ids=train_ids,
                                                            n_segments=n_segments,
                                                            batch_size=batch_size,
                                                            is_augment=True, 
                                                            rotate_clip=True, 
                                                            random_histogram_eq=0.2, 
                                                            random_brightness=(0.5, 2.0), 
                                                            random_intensity=0.2,
                                                            random_flip=0.75,
                                                            model_input_pixel_size=(input_shape[0], input_shape[1]),
                                                            one_hot=True,
                                                            save_segments=False,
                                                            evaluated_path=None,
                                                            exclude_boundary_objs=True,
                                                            boundary_width=1);
                                                            
    """**************************************************************************
    Predict and evaluation in training (Not validation dataset).
    """ 
    test_batch_size = 100;
                                                      
    super_pixel_dataset_test = Super_pixel_seg_dataset(tiff_path, 
                                                       label_path, 
                                                       plot,
                                                       active_positive_class);
                                                       
    super_pixel_dataset_test.prepare_superpixel_dataset(is_train=False,
                                                        load_ids=test_ids,
                                                        n_segments=n_segments,
                                                        batch_size=test_batch_size,
                                                        is_augment=False, 
                                                        model_input_pixel_size=(input_shape[0], input_shape[1]),
                                                        one_hot=False,
                                                        save_segments=True,
                                                        evaluated_path=evaluated_path,
                                                        exclude_boundary_objs=False);                                                   
    
    """**************************************************************************"""
#     model.predict_and_evaluate(test_dataset=super_pixel_dataset_test, \
#                                steps_per_epoch=int(math.ceil(super_pixel_dataset_test.get_n_samples()/float(test_batch_size))))                     
    model.train(generated_dataset=super_pixel_dataset_training,
                epochs=epochs,
                lr=0.01,
                optim="SGD",
                momentum=0.9,
                steps_per_epoch=int(math.ceil(super_pixel_dataset_training.get_n_samples()/float(batch_size))),
                verbose=1,
                loss_fun=loss_fun,
                use_class_weight=False,
                model_save_period=50,
                reduceLr_verbose=1,
                reduceLr_epsilon=0.003,
                reduceLr_factor=0.8,
                reduceLr_minlr=0.00001,
                reduceLr_patience=15,
                reduceLr_cooldown=5,
                predict_in_test=True,
                test_dataset=super_pixel_dataset_test,
                steps_per_epoch_test=int(math.ceil(super_pixel_dataset_test.get_n_samples()/float(test_batch_size))),
                test_period=50,
                test_verbose=1,
                object_based_evaluate=True, 
                pixel_based_evaluate=True);
                          
    print("==================All end!==================");





















