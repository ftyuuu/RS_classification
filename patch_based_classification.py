#coding: utf-8
'''Patch_based分类'''
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

"""=================Tiff ids================="""
train_ids = np.arange(3) + 1;
test_ids = [6];

"""=================Shared parameters================="""
input_shape = (256, 256, 3);
batch_size = 1;
epochs = 1000;
load_weights_path = None;
# class_mode = "binary";
class_mode = "categorical";
# loss_fn = "binary_crossentropy";
loss_fn = "categorical_crossentropy";
loss_fn = "focal_loss_2d"
"""=================Unet parameters================="""
# model_name = "unet";
# model_alias_name = "unet_1";

"""=================RefineNet parameters================="""
model_name = "refinenet"; 
model_alias_name = "refinenet_1";
upSampling2D_Bilinear = True;
chained_res_pool_improved = True;

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
    from utils import Patch_based_dataset;
    from model import ModelBuilder;
    
    plot = Vaihingen_class;
    active_positive_class = [];
    active_positive_class.append(Vaihingen_class.Building);
    active_positive_class.append(Vaihingen_class.Tree);
    active_positive_class.append(Vaihingen_class.Car);
    active_positive_class.append(Vaihingen_class.Low_vegetation);
    classes = len(active_positive_class) + 1;
    
    patch_based_dataset_training = Patch_based_dataset(tiff_path, 
                                                       label_path,
                                                       plot,
                                                       active_positive_class);
    model = ModelBuilder(PAI_FLAGS=FLAGS,
                         input_shape=input_shape,
                         classes=classes,
                         model_name=model_name,
                         model_alias_name=model_alias_name,
                         load_weights=load_weights_path,
                         class_mode=class_mode,
                         upSampling2D_Bilinear=upSampling2D_Bilinear,
                         chained_res_pool_improved=chained_res_pool_improved);    
    
#     from keras.utils.vis_utils import plot_model;
#     plot_model(model.model, to_file="1.png", show_shapes=True);
    
    patch_based_dataset_training.prepare_patch_based_dataset(is_train=True,
                                                             load_ids=train_ids,
                                                             batch_size=batch_size,
                                                             class_mode=class_mode,
                                                             classes=classes,
                                                             is_augment=True, 
                                                             rotate_clip=True, 
                                                             random_histogram_eq=0.2, 
                                                             random_brightness=(0.5, 2.0), 
                                                             random_intensity=0.2,
                                                             random_flip=0.75,
                                                             model_input_pixel_size=(input_shape[0], input_shape[1]),
                                                             evaluated_path=None);
      
    """**************************************************************************
    Predict and evaluation in training (Not validation dataset).
    """ 
    test_batch_size = 8;
                                                         
    patch_based_dataset_test = Patch_based_dataset(tiff_path, 
                                                   label_path, 
                                                   plot,
                                                   active_positive_class);
                                                          
    patch_based_dataset_test.prepare_patch_based_dataset(is_train=False,
                                                         load_ids=test_ids,
                                                         batch_size=test_batch_size,
                                                         is_augment=False, 
                                                         model_input_pixel_size=(input_shape[0], input_shape[1]),
                                                         predict_center_pixel_size=(128, 128),
                                                         evaluated_path=evaluated_path);                                                   
      
#     model.predict_and_evaluate(test_dataset=patch_based_dataset_test, 
#                                steps_per_epoch=int(math.ceil(patch_based_dataset_test.get_n_samples()/float(test_batch_size))), 
#                                verbose=1, 
#                                pixel_based_evaluate=True);
    """**************************************************************************"""
      
    model.train(generated_dataset=patch_based_dataset_training,
                epochs=epochs,
                lr=0.01,
                optim="SGD",
                momentum=0.9,
                steps_per_epoch=int(math.ceil(patch_based_dataset_training.get_n_samples()/float(batch_size))),
                verbose=1,
                loss_fun=loss_fn,
                use_class_weight=False,
                model_save_period=50,
                reduceLr_verbose=1,
                reduceLr_epsilon=0.003,
                reduceLr_factor=0.8,
                reduceLr_minlr=0.00001,
                reduceLr_patience=10,
                reduceLr_cooldown=5,
                predict_in_test=True,
                test_dataset=patch_based_dataset_test,
                steps_per_epoch_test=int(math.ceil(patch_based_dataset_test.get_n_samples()/float(test_batch_size))),
                test_period=1,
                test_verbose=1);
                             
    print("==================All end!==================");


































