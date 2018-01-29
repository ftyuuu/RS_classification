#coding: utf-8
from __future__ import print_function;
from __future__ import division;
from resnet import ResnetBuilder;
from unet import UnetBuilder;
from keras import optimizers;
from keras.callbacks import Callback;
from keras import backend as K;
import tensorflow as tf;
import utils;
import numpy as np;
import os;
import math;
import graph_utils

class PAI_Copy_Callbacks(Callback):
    def __init__(self, copy_path, save_path, save_name):
        super(Callback, self).__init__();
        self.copy_path = copy_path;
        self.save_path = save_path;
        self.save_name = save_name;
        
    def on_epoch_begin(self, epoch, logs={}):
        copy_name = self.save_name + "%d.h5" % (epoch - 1);
        if os.path.exists(os.path.join(self.copy_path, copy_name)):
            
            tf.gfile.Copy(os.path.join(self.copy_path, copy_name), 
                                       os.path.join(self.save_path, copy_name), 
                                       overwrite=True);
            print("Copy weight file successfully.", copy_name);
        else:
            print(os.path.join(self.copy_path, copy_name), " doesn't exists.")    

class ModelBuilder():
    
    def __init__(self, PAI_FLAGS, 
                 input_shape=None, 
                 classes=None, 
                 model_name=None,
                 model_alias_name=None, 
                 load_weights=None, 
                 **kwargs):
        
        if PAI_FLAGS is None:
            self.PAI = False;
        else:
            self.PAI_FLAGS = PAI_FLAGS;
            self.PAI = True;
            
        self.input_shape = input_shape;
        self.model_name = model_name;
        self.model_alias_name = model_alias_name;
        self.classes = classes;
        self.model = None;
        self.load_weights = load_weights;
        self.kwargs = kwargs;
        self.build();
        
    def build(self):
        
        if self.model_name == "resnet":
            original_resnet = self.kwargs.setdefault("original_resnet", False);
            if original_resnet:
                print("Warning: you are using original resnet");
            self.model = ResnetBuilder.build_resnet_18(self.input_shape, self.classes, original_resnet=original_resnet);
        
        if self.model_name == "unet":
            class_mode = self.kwargs.setdefault("class_mode", "categorical");
            self.model = UnetBuilder.build(self.input_shape, [64, 128, 256, 512, 1024], self.classes, class_mode=class_mode);
        
        self.model.summary();
        if self.load_weights is not None:
            print("Loading weights: ", self.load_weights);
            self.model.load_weights(self.load_weights);
        #plot_model(self.model, to_file=self.model_name + ".png", show_shapes=True);
    
    def train(self, 
              generated_dataset=None, 
              epochs=None, 
              lr=0.01,
              optim='Adam',
              momentum=0.9,
              steps_per_epoch=None, 
              verbose=1,
              loss_fun=None,
              use_class_weight=False, 
              model_save_period=100,
              reduceLr_verbose=1,
              reduceLr_epsilon=0.001,
              reduceLr_factor=0.8,
              reduceLr_minlr=0.0001,
              reduceLr_patience=10,
              reduceLr_cooldown=5,
              predict_in_test=True,
              test_dataset=None,
              steps_per_epoch_test=None,
              test_period=50,
              test_verbose=1,
              object_based_evaluate=True, 
              pixel_based_evaluate=True):
        
        """predict_in_test is true means you can predict and evaluate test dataset in training.
        Note: it is not same as validation dataset."""
        assert isinstance(generated_dataset, utils.TiffsDataset);
        assert loss_fun is not None;
        
        if predict_in_test:
            assert isinstance(test_dataset, utils.TiffsDataset);
            assert steps_per_epoch_test is not None;
        
        if use_class_weight:
            tmp_class_weights, classes = generated_dataset.get_class_weight();
            class_weights = {};
            for i in range(len(classes)):
                class_weights[classes[i]] = tmp_class_weights[i];
            print("The class weights is ", class_weights);
        
        """Optimizers"""
        assert optim in ['Adam', 'SGD']
        if optim == 'Adam':
            optimizer = optimizers.Adam(lr=lr);
        elif optim == 'SGD':
            optimizer = optimizers.SGD(lr=lr, momentum=momentum);
        else:
            print("Error--------------Invalid optimizer.");
        
        if loss_fun == "focal_loss_1d":
            self.model.compile(optimizer=optimizer, loss=graph_utils.focal_loss_1d, metrics=["accuracy"]);
        elif loss_fun == "focal_loss_2d":
            self.model.compile(optimizer=optimizer, loss=graph_utils.focal_loss_2d, metrics=["accuracy"]);
        else:
            self.model.compile(optimizer=optimizer, loss=loss_fun, metrics=["accuracy"]);
        
        
        losses = [];
        last_loss = np.inf;
        cur_patience = 0;
        cur_cool_time = np.inf;
        
        for epoch in range(epochs):
            print("epoch: ", epoch+1, "/", epochs);
            epoch_loss = [];
            epoch_acc = [];
            for step in range(steps_per_epoch):
                x, y = generated_dataset.next();
                if use_class_weight:
                    loss_acc = self.model.train_on_batch(x, y, class_weight=class_weights);
                else: 
                    loss_acc = self.model.train_on_batch(x, y);
                
                if verbose:
                    if step % 10 == 0:
                        if self.PAI:
                            print(step+1, "/", steps_per_epoch, " - loss: ", loss_acc[0], "- acc: ", loss_acc[1]);
                        else:
                            pro = int(math.floor((step+1)/float(steps_per_epoch)*30));
                            probar_str = pro * "=" + ">" + (29 - pro) * ".";
                            print(step+1, "/", steps_per_epoch, (4-len(str(step+1)))*" ", \
                                  "[", probar_str, "] - loss: ", loss_acc[0], "- acc: ", loss_acc[1]);
                
                epoch_loss.append(loss_acc[0]);
                epoch_acc.append(loss_acc[1]);
            
            """The mean and acc of losses in one epoch."""
            mean_loss = np.mean(epoch_loss);
            mean_acc = np.mean(epoch_acc);
            
            print("Current epoch ", epoch+1, " - mean_loss:", mean_loss, " - mean_acc:", mean_acc);
            
            if (epoch+1) % model_save_period == 0:
                print(self.model_alias_name + " weights was saved at epoch ", epoch+1);
                save_name = self.model_alias_name + "_weights_epoch" + str(epoch+1) + ".h5";
                self.model.save_weights(os.path.join("./", save_name));
                """For PAI"""
                if self.PAI:
                    tf.gfile.Copy(os.path.join("./", save_name), 
                                  os.path.join(self.PAI_FLAGS.checkpointDir, save_name), 
                                  overwrite=True);
            
            """Whether to change learning rate."""    
            losses.append(mean_loss);
            if cur_cool_time > reduceLr_cooldown:
                if mean_loss + reduceLr_epsilon >= last_loss:
                    cur_patience = cur_patience + 1;
                    if cur_patience > reduceLr_patience:
                        curlr = K.get_value(self.model.optimizer.lr);
                        reducedlr = curlr * reduceLr_factor;
                        if reducedlr >= reduceLr_minlr:
                            if reduceLr_verbose:
                                print("==========Learning rate reduce from ", curlr, " to ", reducedlr, "==========");
                            K.set_value(self.model.optimizer.lr, reducedlr);
                            cur_cool_time = 0;
                            cur_patience = 0;
                else:
                    last_loss = mean_loss;
                    cur_patience = 0;
                    
            cur_cool_time = cur_cool_time + 1;
            
            """#################################################################################"""
            if (epoch+1) % test_period == 0 and predict_in_test:
                self.predict_and_evaluate(test_dataset=test_dataset, 
                                          steps_per_epoch=steps_per_epoch_test, 
                                          verbose=test_verbose, 
                                          object_based_evaluate=object_based_evaluate, 
                                          pixel_based_evaluate=pixel_based_evaluate);
            """#################################################################################"""
            
            
            if (epoch+1) % 10 == 0:
                print("==========Current learning rate is ", K.get_value(self.model.optimizer.lr));
            print("-------------------------------------------------------");
        
        save_name = self.model_alias_name + "_weights_epoch" + str(epoch+1) + ".h5";
        
        print("==================train end!==================");
        """For PAI"""
        if self.PAI:
            self.model.save_weights(os.path.join("./", save_name));
            tf.gfile.Copy(os.path.join("./", save_name), 
                          os.path.join(self.PAI_FLAGS.checkpointDir, save_name), 
                          overwrite=True);
        else:
            self.model.save_weights(save_name);
            
        return losses;
            
    def predict_and_evaluate(self, 
                             test_dataset=None, 
                             steps_per_epoch=None, 
                             verbose=1, 
                             object_based_evaluate=True, 
                             pixel_based_evaluate=True,
                             show_class_result_pic=False):
        
        """In some dataset, the evaluated mask is not same as labeld mask."""
        assert isinstance(test_dataset, utils.TiffsDataset);
        assert test_dataset.is_train == False;
        
        if show_class_result_pic:
            assert pixel_based_evaluate == True;
        
        if test_dataset.is_augment == True:
            print("Warning: you are evaluating in data augmentation!");
        
        evaluated_buff_class = test_dataset.plot.EvaluatedBuff.value[1];
        
        if isinstance(test_dataset, utils.Super_pixel_seg_dataset):
            obj_y_trues, obj_y_preds, obj_y_probs, pixel_y_trues, pixel_y_preds, pixel_y_probs, class_result_pics = \
                self.obic_predict(test_dataset, steps_per_epoch=steps_per_epoch, pixel_based_evaluate=pixel_based_evaluate, \
                                  verbose=verbose, evaluated_buff_class=evaluated_buff_class, show_class_result_pic=show_class_result_pic);
        
            if object_based_evaluate:
                self.print_evalation(obj_y_trues, obj_y_preds, obj_y_probs, test_dataset, "Object-based");
            
            if pixel_based_evaluate:
                self.print_evalation(pixel_y_trues, pixel_y_preds, pixel_y_probs, test_dataset, "Pixel-based");
            
        if isinstance(test_dataset, utils.Patch_based_dataset):
            pixel_y_trues, pixel_y_preds, pixel_y_probs, class_result_pics = \
                self.pbic_predict(test_dataset, steps_per_epoch=steps_per_epoch, verbose=verbose, \
                                  evaluated_buff_class=evaluated_buff_class, show_class_result_pic=show_class_result_pic);
                                  
            self.print_evalation(pixel_y_trues, pixel_y_preds, pixel_y_probs, test_dataset, "Pixel-based");
            
        if show_class_result_pic:
            return class_result_pics;
    
    def obic_predict(self, 
                    obic_dataset_test, 
                    steps_per_epoch=None, 
                    pixel_based_evaluate=True, 
                    verbose=1,
                    evaluated_buff_class=None,
                    show_class_result_pic=False):
        
        if pixel_based_evaluate:
            assert obic_dataset_test.save_segments, "If performing the pixel-based evaluation, the save_segments must be true.";
        
        y_preds = [];
        y_probs = [];
        y_trues = [];
        pic_ids = [];
        fids = [];
        
        for step in range(steps_per_epoch):
            x, y_true, pic_id, fid = obic_dataset_test.next();
            y_trues.extend(y_true);
            pic_ids.extend(pic_id);
            fids.extend(fid);
            y_pred = self.model.predict_on_batch(x);
            y_preds.extend(np.argmax(y_pred, axis=1));
            if obic_dataset_test.classes == 2:
                y_probs.extend(y_pred[:, 1]);
            
            if verbose == 1:
                if (step+1) % 10 == 0:
                    print("Predicting... - ", (step+1), "/", steps_per_epoch);
        
        y_preds = np.array(y_preds).flatten();
        y_trues = np.array(y_trues).flatten();
        pic_ids = np.array(pic_ids).flatten();
        fids = np.array(fids).flatten();
        
        if len(y_probs) > 0:
            y_probs = np.array(y_probs).flatten();
            dt = np.array(list(zip(pic_ids, fids, y_trues, y_preds, y_probs)));
        else:
            dt = np.array(list(zip(pic_ids, fids, y_trues, y_preds)));
        del pic_ids; del fids; del y_trues; del y_preds; del y_probs;
        
        """Based on fids, writting predicted results to object's pixels."""
        class_result_pics = {};
        prob_result_pics = {};
        if pixel_based_evaluate:
            for cur_pic_id in obic_dataset_test.load_ids:
                cur_class_result = np.copy(obic_dataset_test.segments[cur_pic_id]).astype(np.float32);
                if obic_dataset_test.classes == 2:
                    cur_prob_result = np.copy(obic_dataset_test.segments[cur_pic_id]).astype(np.float32);
                
                cur_dt = dt[dt[:, 0]==cur_pic_id, :];
                for item in cur_dt:
                    fid = item[1];
                    y_pred = item[3];
                    cur_class_result[cur_class_result == fid] = y_pred;
                    if obic_dataset_test.classes == 2:
                        cur_prob_result[cur_prob_result == fid] = item[4];
                    
                class_result_pics[cur_pic_id] = cur_class_result.astype(np.int32);
                if obic_dataset_test.classes == 2:
                    prob_result_pics[cur_pic_id] = cur_prob_result;
                del cur_dt;
        
        y_preds = [];
        y_probs = [];
        y_trues = [];
        for pic_id in class_result_pics.keys():
            y_preds.extend(class_result_pics[pic_id].flatten());
            if obic_dataset_test.classes == 2:
                y_probs.extend(prob_result_pics[pic_id].flatten());
            y_trues.extend(obic_dataset_test.evaluations[pic_id][:, :, 0].flatten());
        
        if obic_dataset_test.classes == 2:
            y = np.array(list(zip(y_trues, y_preds, y_probs)));
        else:
            y = np.array(list(zip(y_trues, y_preds)));
        del y_trues; del y_preds; del y_probs;
        
        """Exclude evaluated buff from evaluation."""
        if evaluated_buff_class is not None:
            dt = dt[dt[:, 2] != evaluated_buff_class, :];  
            y = y[y[:, 0] != evaluated_buff_class, :];
        print("OBIC-After excluding evaluated buff, the count of segmented objects: ", dt.shape[0]);
        
        if show_class_result_pic:
            if obic_dataset_test.classes == 2:
                return dt[:, 2], dt[:, 3], dt[:, 4], y[:, 0], y[:, 1], y[:, 2], class_result_pics;
            else:
                return dt[:, 2], dt[:, 3], None, y[:, 0], y[:, 1], None, class_result_pics;
        else:
            if obic_dataset_test.classes == 2:
                return dt[:, 2], dt[:, 3], dt[:, 4], y[:, 0], y[:, 1], y[:, 2], None;
            else:
                return dt[:, 2], dt[:, 3], None, y[:, 0], y[:, 1], None, None;
            
    def pbic_predict(self, 
                     pbic_dataset_test=None, 
                     steps_per_epoch=None, 
                     verbose=1,
                     evaluated_buff_class=None,
                     show_class_result_pic=False):
        #y_preds = {};
        #pic_ids = [];
        y_preds = {};
        
        for pic_id in pbic_dataset_test.load_ids:
#             y_preds[pic_id] = np.zeros([super_pixel_dataset_test.evaluations[pic_id].shape[0],
#                                         super_pixel_dataset_test.evaluations[pic_id].shape[1],
#                                         4]);
            channels = pbic_dataset_test.classes;
            if pbic_dataset_test.class_mode == 'binary':
                channels -= 1;
            y_preds[pic_id] = np.zeros([pbic_dataset_test.evaluations[pic_id].shape[0],
                                        pbic_dataset_test.evaluations[pic_id].shape[1],
                                        channels]);
            
        for step in range(steps_per_epoch):
            #x, pic_id, loc_id, cur_centroids = super_pixel_dataset_test.next();
            #pic_ids.extend(cur_pic_ids);
            x, cur_pic_ids, cur_centroids = pbic_dataset_test.next();
            y_pred = self.model.predict_on_batch(x);
            y_pred = utils._get_center_img(y_pred, pbic_dataset_test.predict_center_pixel_size);
            r_begins, r_ends, c_begins, c_ends = utils._calculate_extent_by_centers(cur_centroids, pbic_dataset_test.predict_center_pixel_size);
            
            cur_tif_ids = np.unique(cur_pic_ids);
            for i in cur_tif_ids:
                inx = (cur_pic_ids == i);
                cur_pred = y_pred[inx];
                #cur_loadid = loc_id[inx];
                img_r_begins_new, img_r_ends_new, img_c_begins_new, img_c_ends_new, \
                extent_r_begins_new,  extent_r_ends_new, extent_c_begins_new, extent_c_ends_new = \
                    utils._align_extent_and_return_intersected_extents(np.array(zip(r_begins[inx], r_ends[inx], \
                                                                                    c_begins[inx], c_ends[inx])), y_preds[i].shape[:2]);
                
                for j in range(np.sum(inx)):
                    #y_preds[pic_ids[i]][img_r_begins_new[j]:img_r_ends_new[j], img_c_begins_new[j]:img_c_ends_new[j], cur_loadid[j]] = \
                        #cur_pred[j, extent_r_begins_new[j]:extent_r_ends_new[j], extent_c_begins_new[j]:extent_c_ends_new[j], 0];
                    y_preds[i][img_r_begins_new[j]:img_r_ends_new[j], img_c_begins_new[j]:img_c_ends_new[j], :] += \
                        cur_pred[j, extent_r_begins_new[j]:extent_r_ends_new[j], extent_c_begins_new[j]:extent_c_ends_new[j], :];
            
            #a = np.mean(y_preds[1], axis=2);
            #b = y_preds222[1]/4.;
            #assert np.sum(a == b) == 4929911;
            if verbose == 1:
                if (step+1) % 10 == 0:
                    print("Predicting... - ", (step+1), "/", steps_per_epoch);
        
        y_preds_all = [];
        y_probs_all = [];"""For ap metrics (binary)"""
        y_trues_all = [];
        class_result_pics = {};
        
        for cur_pic_id in pbic_dataset_test.load_ids:
            y_preds[cur_pic_id] = y_preds[cur_pic_id] / 4.;
            if y_preds[cur_pic_id].shape[-1] == 1:
                y_probs_all.extend(y_preds[cur_pic_id][:, :, 0].flatten());
                tmp_class = np.round(y_preds[cur_pic_id])[:, :, 0];
            else:   
                tmp_class = np.argmax(y_preds[cur_pic_id], axis=-1);
            y_preds_all.extend(tmp_class.flatten());
            if show_class_result_pic:
                class_result_pics[cur_pic_id] = tmp_class;
               
            y_trues_all.extend(pbic_dataset_test.evaluations[cur_pic_id][:, :, 0].flatten());
        
        y_probs_all = np.array(y_probs_all);
        y_trues_all = np.array(y_trues_all);
        y_preds_all = np.array(y_preds_all);
        
        if evaluated_buff_class is not None:
            inx = y_trues_all != evaluated_buff_class;
            if y_probs_all.shape[0] > 0:
                y_probs_all = y_probs_all[inx];
            y_trues_all = y_trues_all[inx]; 
            y_preds_all = y_preds_all[inx];
        print("PBIC-After excluding evaluated buff, the count of pxiels: ", y_trues_all.shape[0]);
        
        if show_class_result_pic:
            return y_trues_all, y_preds_all, y_probs_all, class_result_pics;
        else:
            return y_trues_all, y_preds_all, y_probs_all, None;
    
    @staticmethod
    def print_evalation(y_trues_all, y_preds_all, y_probs_all, test_dataset, mode):
        assert mode in ["Object-based", "Pixel-based"];
        
        picids = test_dataset.load_ids;
        labels = [];
        labels_name = [];
        labels.append(test_dataset.plot.Background.value[1]);
        labels_name.append(test_dataset.plot.Background.value[2]);
        for item in test_dataset.active_positive_class:
            if item != test_dataset.plot.EvaluatedBuff:
                _, label, label_name = item.value;
                labels.append(label);
                labels_name.append(label_name);
            
        oa, pre, recall, f1, kappa, ap_or_confmat = \
            utils._evaluate_acc(y_trues_all, y_preds_all, y_probs_all, labels=labels);
        
        print("===================Calculate the accuracy of segmented objects===================");
        print("%s evalation. The accuracy_score of pic "%mode, picids, ": \n", oa);
        if not (isinstance(pre, list)) or not (isinstance(recall, list)):
            str = "ap";
        else:
            str = "confusion matrix";
            labels_name.append("Weighted ave");
            
            pre = zip(labels_name, pre);
            recall = zip(labels_name, recall);
            f1 = zip(labels_name, f1);
            
            ap_or_confmat = np.pad(ap_or_confmat, ((1, 0), (1, 2)), 'constant').astype(np.float);
            ap_or_confmat[1:, -1] =  np.sum(ap_or_confmat, axis=1)[1:];
            j = 1;
            for i in range(1, ap_or_confmat.shape[0]):
                ap_or_confmat[i, -2] =  ap_or_confmat[i, j] / ap_or_confmat[i, -1];
                j += 1;
            ap_or_confmat = ap_or_confmat.astype(np.str);
            
            labels_name.pop();
            ap_or_confmat[1:, 0] = ['Act: ' + i for i in labels_name];
            labels_name = ['| Pre: '+ i for i in labels_name];
            labels_name.append("Acc");labels_name.append("Sum");
            ap_or_confmat[0, 1:] = labels_name;
            ap_or_confmat = print_as_table(ap_or_confmat);
        
        print("%s evalation. The precision of pic "%mode, picids, ": \n", pre);
        print("%s evalation. The recall of pic "%mode, picids, ": \n", recall);
        print("%s evalation. The f1 score of pic "%mode, picids, ": \n", f1);
        print("%s evalation. The kappa of pic "%mode, picids, ": \n", kappa);
        print("%s evalation. The %s of pic "%(mode, str), picids, ": \n", ap_or_confmat);   
            
def print_as_table(arr):
    out = '';
    header = arr[0, :].tolist()[1:];
    head_fmt = '{:>{width}} ' + ' {:^17}' * len(header);
    out = head_fmt.format('', *header, width=20);
    out += '\n';
    
    row_fmt = '{:>{width}} ' + '{:<25}' + '{:^18.{digit}f}'*(arr.shape[1]-1);
    for i in range(1, arr.shape[0]):
        row = arr[i, :].tolist();
        row_header = row[0];
        row_values = map(float, row[1:]);
        out += row_fmt.format('', row_header, *row_values, width=0, digit=2);
        out += '\n';
    return out;



