#coding: utf-8
"""预测训练的模型"""
from __future__ import print_function;
from utils import Vaihingen_class;
from utils import Super_pixel_seg_dataset;
from utils import Patch_based_dataset;
from model import ModelBuilder;
from matplotlib import pyplot as plt;
from keras import losses
import math;

tiff_path = "/media/fty/Windows/linux_data/PAI_Demo/data/";
label_path = "/media/fty/Windows/linux_data/PAI_Demo/label/";
evaluated_path = "/media/fty/Windows/linux_data/PAI_Demo/evaluation/";

def patch_batch_classification_predict():
    test_ids = [6];

    input_shape = (256, 256, 3);
    
    #class_mode = "binary";
    class_mode = "categorical";
    
    model_name = "refinenet";
    #model_weights = "/media/fty/Windows/linux_data/weights/binary/pbic/unet_weights_epoch100.h5";
    model_weights = "/media/fty/Windows/linux_data/weights/categorical/refinement/refinenet_weights_epoch300.h5";
    
    plot = Vaihingen_class;
    active_positive_class = [];
    active_positive_class.append(Vaihingen_class.Building);
    active_positive_class.append(Vaihingen_class.Tree);
    active_positive_class.append(Vaihingen_class.Car);
    active_positive_class.append(Vaihingen_class.Low_vegetation);
    
    classes = len(active_positive_class) + 1;
    
    test_batch_size = 8;
    patch_based_dataset_test = Patch_based_dataset(tiff_path, 
                                                   label_path, 
                                                   plot,
                                                   active_positive_class);
                                                        
    patch_based_dataset_test.prepare_patch_based_dataset(is_train=False,
                                                        load_ids=test_ids,
                                                        batch_size=test_batch_size,
                                                        class_mode=class_mode,
                                                        classes=classes,
                                                        is_augment=False, 
                                                        model_input_pixel_size=(input_shape[0], input_shape[1]),
                                                        predict_center_pixel_size=(128, 128),
                                                        evaluated_path=evaluated_path);
    model = ModelBuilder(PAI_FLAGS=None,
                         input_shape=input_shape,
                         classes=classes,
                         model_name=model_name,
                         load_weights=model_weights,
                         class_mode=class_mode);
                         
    class_result_pics = model.predict_and_evaluate(test_dataset=patch_based_dataset_test, 
                                                   steps_per_epoch=int(math.ceil(patch_based_dataset_test.get_n_samples()/float(test_batch_size))), 
                                                   verbose=1, 
                                                   pixel_based_evaluate=True,
                                                   show_class_result_pic=True);
    
    for i in class_result_pics.keys():
        plt.subplot(121);
        plt.imshow(patch_based_dataset_test.evaluations[i][:, :, 0]);
        plt.subplot(122);
        plt.imshow(class_result_pics[i]);
        plt.show();
    
def super_pixel_classification_predict():
    
    test_ids = [6];
    
    input_shape = (64, 64, 3);
    n_segments = 20;
    model_name = "resnet";
    
    model_weights = "/media/fty/Windows/linux_data/weights/categorical/resnet/resnet_super_pixel_weights_epoch450.h5";
    
    plot = Vaihingen_class;
    active_positive_class = [];
    active_positive_class.append(Vaihingen_class.Building);
    active_positive_class.append(Vaihingen_class.Tree);
    active_positive_class.append(Vaihingen_class.Car);
    active_positive_class.append(Vaihingen_class.Low_vegetation);
    classes = len(active_positive_class) + 1;
    
    super_pixel_dataset_test = Super_pixel_seg_dataset(tiff_path, 
                                                       label_path, 
                                                       plot,
                                                       active_positive_class);
    
    test_batch_size = 100;                                                  
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
                                                        
    model = ModelBuilder(PAI_FLAGS=None,
                         input_shape=input_shape, 
                         classes=classes, 
                         model_name=model_name,
                         load_weights=model_weights);
    
    class_result_pics = model.predict_and_evaluate(test_dataset=super_pixel_dataset_test, 
                                                   steps_per_epoch=int(math.ceil(super_pixel_dataset_test.get_n_samples()/float(test_batch_size))),
                                                   verbose=1, 
                                                   object_based_evaluate=True, 
                                                   pixel_based_evaluate=True,
                                                   show_class_result_pic=True)                     
    
#     print(class_result_pics);
#     print(prob_result_pics);
#     print(class_result_pics[6].shape);
#     print(prob_result_pics[6].shape);
    
    for i in class_result_pics.keys():
        plt.subplot(121);
        plt.imshow(super_pixel_dataset_test.evaluations[i][:, :, 0]);
        plt.subplot(122);
        plt.imshow(class_result_pics[i]);
        
        plt.show();
    
if __name__ == '__main__':    
    #super_pixel_classification_predict();
    patch_batch_classification_predict();    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
                                                        