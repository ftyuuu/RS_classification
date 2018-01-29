#coding: utf-8
"""Utils工具"""
from __future__ import print_function;
from __future__ import division;
from enum import Enum;
from functools import partial;
from skimage import io, color, transform, exposure, img_as_float;
from skimage.measure import label;
from sklearn.utils import compute_class_weight;
from sklearn import metrics;
from keras.utils import to_categorical;
import numpy as np;
import os;
import random;
import math;
import multiprocessing;

#####################################################
# Image processing functions
##################################################### 
def _random_rotate_img(image, max_angle=180, rotate=None, prob=0.8):
    """image: [row, col, channel]
    """
    if rotate == None:
        r = random.uniform(0, 1);
        if r >= 1 - prob:
            r = random.uniform(0, max_angle);
        else:
            r = 0;
    else:
        assert np.all([rotate >= 0, rotate <= 180]), """rotate must in range [0, 180]""";
        r = rotate;
    return np.where(r != 0., transform.rotate(image, r), image), r;

def _random_histogram_eq_img(image, prob=0.5):
    """image: [row, col, channel]
    """
    r = random.uniform(0, 1);
    if r >= 1 - prob:
        image1 = np.copy(image);
        image1 = color.rgb2hsv(image1);
        image1[:, :, 2] = exposure.equalize_hist(image1[:, :, 2]);
        image1 = color.hsv2rgb(image1);
        return image1;
    else:
        return image;

def _random_brightness_img(image, min=0.5, max=2.0, prob=0.9):
    """image: [row, col, channel]
    """
    r = random.uniform(0, 1);
    if r >= 1 - prob:
        r = random.uniform(min, max);
    else:
        r = 1.0;
    return np.where(r != 1.0, exposure.adjust_gamma(image, r), image);

def _random_intensity_img(image, prob=0.5):
    """image: [row, col, channel]
    """
    r = random.uniform(0, 1);
    return np.where(r >= 1 - prob, exposure.rescale_intensity(image, out_range=(0, 1)), image);

def _random_flip_img(image, prob=0.75, flip_mode=None):
    """image: [row, col, channel]
    """
    image1 = np.copy(image);
    r = random.uniform(0, 1);
    
    if flip_mode is None:
        if r <= 1 - prob:
            flip_mode = "unchange";
        else:
            r = ["ud", "lr", "ud_lr"];
            np.random.shuffle(r);
            flip_mode = r[0];

    assert flip_mode in ["ud", "lr", "ud_lr", "unchange"], "Unknown flip_mode";
    if flip_mode == "ud":
        image1 = np.flipud(image1);
    elif flip_mode == "lr":
        image1 = np.fliplr(image1);
    elif flip_mode == "ud_lr":
        image1 = np.fliplr(np.flipud(image1));
    else:
        pass;
    return image1, flip_mode;

def _clip_image(image, centroids, crop_size):
    
    img_r, img_c = image.shape[0], image.shape[1];
    assert centroids.dtype == np.int, """Centroids must be integer""";
    
    r_begins, r_ends, c_begins, c_ends = \
        _calculate_extent_by_centers(centroids, crop_size);
    
    r_begins_new, r_ends_new, c_begins_new, c_ends_new, \
    r_begins_shift, r_ends_shift, c_begins_shift, c_ends_shift = \
        _align_extent_and_return_shift(np.array(zip(r_begins, r_ends, c_begins, c_ends)), (img_r, img_c));
    
    return_images = [image[rb:re, cb:ce] for rb, re, cb, ce in 
                     zip(r_begins_new, r_ends_new, c_begins_new, c_ends_new)];
    return_images = np.array([np.pad(img, [(rb, re), (cb, ce), (0, 0)], "constant", constant_values=0) for rb, re, cb, ce, img in 
                     zip(r_begins_shift, r_ends_shift, c_begins_shift, c_ends_shift, return_images)]);
    
    assert len(return_images.shape) > 1, "The shape of img in return images is not same.";
    return return_images;

def _random_rotate_clip_image(image, centroids, crop_size, rotate_seed=None):
    
    img_r, img_c = image.shape[0], image.shape[1];
    assert centroids.dtype == np.int, """Centroids must be integer""";
    
    r, c = centroids[:, 0], centroids[:, 1];
    assert np.all([r >= 0, r <= img_r - 1,  c >= 0, c <= img_c - 1]), "Row and column are out of image";
        
    s1, s2 = crop_size;
    diag_l_half = int(math.ceil(np.sqrt(s1 * s1 + s2 * s2)/2.));
    
    r_begins = r - diag_l_half;
    r_ends = r + diag_l_half + 1;
    c_begins = c - diag_l_half;
    c_ends = c + diag_l_half + 1;
    
    r_begins_new, r_ends_new, c_begins_new, c_ends_new, \
    r_begins_shift, r_ends_shift, c_begins_shift, c_ends_shift = \
        _align_extent_and_return_shift(np.array(zip(r_begins, r_ends, c_begins, c_ends)), (img_r, img_c));
     
    return_images = [image[rb:re, cb:ce] for rb, re, cb, ce in 
                     zip(r_begins_new, r_ends_new, c_begins_new, c_ends_new)];
    return_images = np.array([np.pad(img, [(rb, re), (cb, ce), (0, 0)], "constant", constant_values=0) for rb, re, cb, ce, img in 
                     zip(r_begins_shift, r_ends_shift, c_begins_shift, c_ends_shift, return_images)]);
    
    if rotate_seed is None:
        return_images = batch_operation([return_images], opea_func=_random_rotate_img);
    else:
        return_images = batch_operation([return_images, np.zeros(return_images.shape[0]), np.array(rotate_seed), np.ones(return_images.shape[0])], \
                                        opea_func=_random_rotate_img);

    rotate_seed =  return_images[:, 1];
    return_images = _get_center_img(np.array(list(return_images[:, 0])), crop_size);
    return return_images, rotate_seed;

#####################################################
# Image calculation functions
##################################################### 
def _calculate_center(arr):
    """arrs: [row, col]
    """
    r, c = arr.shape;
    return int(r/2.), int(c/2.);

def _calculate_extent_by_centers(centers, size):
    """centers: [batch, (r_ids, c_ids)]
       size: (rows, cols)
    """
    center_rs = centers[:, 0];
    center_cs = centers[:, 1];
    r_begins = center_rs - int(size[0]/2.);
    r_ends = r_begins + size[0];
    c_begins = center_cs - int(size[1]/2.);
    c_ends = c_begins + size[1];
    
    return r_begins, r_ends, c_begins, c_ends;

def _get_center_img(imgs, size):
    """ imgs: [batch, row, col, channel]
        size: (rows, cols)
    """
    r, c = imgs.shape[1], imgs.shape[2];
    assert r >= size[0] and c >= size[1], """Size must small than img's shape.""";
    
    center_r, center_c = _calculate_center(imgs[0, :, :, 0]);
    r_begins, r_ends, c_begins, c_ends = \
        _calculate_extent_by_centers(np.expand_dims([center_r, center_c], axis=0), size);
    
    return imgs[:, r_begins[0]:r_ends[0], c_begins[0]:c_ends[0], :];

def _align_extent_and_return_intersected_extents(extents, img_shape):
    """extent: [batch, (r_begin, r_end, c_begin, c_end)]
       img_shape: (rows, cols)
    """
    img_r_begins_new, img_r_ends_new, img_c_begins_new, img_c_ends_new, \
    img_r_begins_shift, img_r_ends_shift, img_c_begins_shift, img_c_ends_shift = \
        _align_extent_and_return_shift(extents, img_shape);

    extent_r_begins_new = img_r_begins_shift;
    extent_r_ends_new = np.where(img_r_ends_shift > 0, extents[:, 1]- extents[:, 0] - img_r_ends_shift, \
                                 extents[:, 1]- extents[:, 0]);
    extent_c_begins_new = img_c_begins_shift;
    extent_c_ends_new = np.where(img_c_ends_shift > 0, extents[:, 3]- extents[:, 2] - img_c_ends_shift, \
                                 extents[:, 3]- extents[:, 2]);
    
    return img_r_begins_new, img_r_ends_new, img_c_begins_new, img_c_ends_new, \
        extent_r_begins_new,  extent_r_ends_new, extent_c_begins_new, extent_c_ends_new;
    
def _align_extent_and_return_shift(extents, img_shape):
    """extents: [batch, (r_begin, r_end, c_begin, c_end)]
       img_shape: (rows, cols)
    """
    img_r, img_c = img_shape;
    r_begins = extents[:, 0];
    r_ends = extents[:, 1];
    c_begins = extents[:, 2];
    c_ends = extents[:, 3];
    
    r_begins_new = np.where(r_begins < 0, 0, r_begins);
    r_ends_new = np.where(r_ends > img_r, img_r, r_ends);
    c_begins_new = np.where(c_begins < 0, 0, c_begins);
    c_ends_new = np.where(c_ends > img_c, img_c, c_ends);
    
    r_begins_shift = np.where(r_begins < 0, abs(r_begins), 0);
    r_ends_shift = np.where(r_ends > img_r, r_ends - img_r, 0);
    c_begins_shift = np.where(c_begins < 0, abs(c_begins), 0);
    c_ends_shift = np.where(c_ends > img_c, c_ends - img_c, 0);
    
    return r_begins_new, r_ends_new, c_begins_new, c_ends_new, \
        r_begins_shift, r_ends_shift, c_begins_shift, c_ends_shift;

#####################################################
# Business function
##################################################### 
def extractMaskByRGBs(img, class_infos):
    """Extract labeled image by rgb colors"""
    row, col, channel = img.shape;
    img = np.reshape(img, [row*col, channel]);
    
    indices = [];
    for class_info in class_infos:
        rgb = class_info.value[0];
        index = np.all([img[:, 0] == rgb[0], img[:, 1] == rgb[1], img[:, 2] == rgb[2]], axis=0);
        indices.append(index);
    
    indices = np.any(indices, axis=0);
    img[(indices == 0), :] = 255;
    return np.reshape(img, [row, col, channel]);

def rgb2LabeledMask(rgmImg, class_infos):
    """Change rgb class mask to 0, 1, 2... mask"""
    row, col, channel = rgmImg.shape;
    img = np.reshape(rgmImg, [row*col, channel]);
    labeled_mask = np.zeros(row*col);
    
    for class_info in class_infos:
        rgb, label, _ = class_info.value;
        index = np.all([img[:, 0] == rgb[0], img[:, 1] == rgb[1], img[:, 2] == rgb[2]], axis=0);
        labeled_mask[index] = label;
    
    return np.reshape(labeled_mask, [row, col]);

def tif2jpg(tifPath, jpgPath):
    img = io.imread(tifPath);
    io.imsave(jpgPath, img);
    
def tifs2jpgs(tifPath, jpgPath):
    for subdir in sorted(os.listdir(tifPath)):
        cur_class_path = os.path.join(tifPath, subdir);
        os.mkdir(os.path.join(jpgPath, subdir));
        
        tifPaths = [os.path.join(cur_class_path, curpic) for curpic in os.listdir(cur_class_path)];
        jpgPaths = [os.path.join(jpgPath, subdir, curpic.split('.')[0] + ".jpg") for curpic in os.listdir(cur_class_path)]
        
        batch_operation([tifPaths, jpgPaths], tif2jpg);

def _evaluate_acc(y_trues, y_preds, y_probs, labels=None):
    if np.unique(y_trues).shape[0] == 2:
        oa = metrics.accuracy_score(y_trues, y_preds);
        pre = metrics.precision_score(y_trues, y_preds, average='binary');
        recall = metrics.recall_score(y_trues, y_preds, average='binary');
        f1 = metrics.f1_score(y_trues, y_preds, average='binary');
        kappa = metrics.cohen_kappa_score(y_trues, y_preds);
        ap = metrics.average_precision_score(y_trues, y_probs);
        return oa, pre, recall, f1, kappa, ap;
    elif np.unique(y_trues).shape[0] > 2:
        oa = metrics.accuracy_score(y_trues, y_preds);
        pre = metrics.precision_score(y_trues, y_preds, labels=labels, average=None).tolist();
        pre.append(metrics.precision_score(y_trues, y_preds, average="weighted"));
        recall = metrics.recall_score(y_trues, y_preds, labels=labels, average=None).tolist();
        recall.append(metrics.recall_score(y_trues, y_preds, average="weighted"));
        confusion_mat = metrics.confusion_matrix(y_trues, y_preds, labels=labels);
        f1 = metrics.f1_score(y_trues, y_preds, labels=labels, average=None).tolist();
        f1.append(metrics.f1_score(y_trues, y_preds, average="weighted"));
        kappa = metrics.cohen_kappa_score(y_trues, y_preds);
        return oa, pre, recall, f1, kappa, confusion_mat;
    else:
        raise "Error: y_tures only have one class.";
    
def gen_pbic_test_data(load_ids, random_range_each_tiff, predict_center_pixel_size):
    ret_centroids = {};
    for i in load_ids:

        cur_tif_row, cur_tif_col = random_range_each_tiff[i];
        
        r_iter = int(math.ceil(cur_tif_row / float(predict_center_pixel_size[0])));
        c_iter = int(math.ceil(cur_tif_col / float(predict_center_pixel_size[1])));
        
        centroids = [];
        """Top left"""
        c_rs1 = [k * predict_center_pixel_size[0] + int(predict_center_pixel_size[0] / 2.) for k in range(r_iter)];
        c_cs1 = [k * predict_center_pixel_size[1] + int(predict_center_pixel_size[1] / 2.) for k in range(c_iter)];
        for j in range(len(c_rs1)):
            #centroids.extend(list(zip([i]*len(c_cs1), [c_rs1[j]]*len(c_cs1), c_cs1, [0]*len(c_cs1))));
            centroids.extend(list(zip([i]*len(c_cs1), [c_rs1[j]]*len(c_cs1), c_cs1)));
        
        """Top right"""
        c_cs2 = [cur_tif_col - int((k + 1/2.) * predict_center_pixel_size[1]) for k in range(c_iter)];
        c_cs2.sort();
        for j in range(len(c_rs1)):
            #centroids.extend(list(zip([i]*len(c_cs2), [c_rs1[j]]*len(c_cs2), c_cs2, [1]*len(c_cs2))));
            centroids.extend(list(zip([i]*len(c_cs2), [c_rs1[j]]*len(c_cs2), c_cs2)));
            
        """Bottom left"""
        c_rs3 = [cur_tif_row - int((k + 1/2.) * predict_center_pixel_size[0]) for k in range(r_iter)];
        c_rs3.sort()
        for j in range(len(c_rs3)):
            #centroids.extend(list(zip([i]*len(c_cs1), [c_rs3[j]]*len(c_cs1), c_cs1, [2]*len(c_cs1))));
            centroids.extend(list(zip([i]*len(c_cs1), [c_rs3[j]]*len(c_cs1), c_cs1)));
            
        """Bottom right"""
        for j in range(len(c_rs3)):
            #centroids.extend(list(zip([i]*len(c_cs2), [c_rs3[j]]*len(c_cs2), c_cs2, [3]*len(c_cs2))));
            centroids.extend(list(zip([i]*len(c_cs2), [c_rs3[j]]*len(c_cs2), c_cs2)));
        assert len(centroids) == r_iter * c_iter * 4;   
        
        ret_centroids[i] = np.array(centroids);
    
    ret_centroids = np.concatenate(list(ret_centroids.values()));   
    return ret_centroids ;
        
def batch_operation(inputs, opea_func):
    return np.array(map(opea_func, *inputs));

#####################################################
# Dataset
#####################################################
class TiffsDataset():
     
    def __init__(self, tiff_path, label_path, plot, active_positive_class):
        
        self.tiff_path = tiff_path;
        self.label_path = label_path;
        self.evaluated_path = None;
        
        self.load_ids = None;
        self.plot = plot;
        self.active_positive_class = active_positive_class;
        self.classes = len(active_positive_class) + 1;
        
        self.tiffs = {};
        self.labeled_mask = {};
        self.evaluations = {};
        
    def load_tiffs(self, load_ids):
        
        self.tiffs = {};
        
        for i in load_ids:
            self.tiffs[i] = img_as_float(io.imread(os.path.join(self.tiff_path, str(i) + ".tif")));
            print("tiff", i, self.tiffs[i].shape);
        
    def load_labels(self, load_ids):
        
        self.labeled_mask = {};
        
        for i in load_ids:
            self.labeled_mask[i] = io.imread(os.path.join(self.label_path, str(i) + "_label.tif"));
            print("label", i, self.labeled_mask[i].shape);
    
    def load_evaluation(self, load_ids, evaluated_path):
        
        self.evaluated_path = evaluated_path;
        self.evaluations = {};
        
        for i in load_ids:
            self.evaluations[i] = io.imread(os.path.join(self.evaluated_path, str(i) + "_evaluation.tif"));
            print("evaluation", i, self.evaluations[i].shape);
    
    def prepare_dataset(self, load_ids):
        
        self.load_ids = load_ids;
        
        """load tiffs"""
        self.load_tiffs(self.load_ids);
        
        """load and modify labeled mask"""
        self.load_labels(self.load_ids);
        
        """check if a and b correspond"""
        for i in self.load_ids:
            assert self.tiffs[i].shape == self.labeled_mask[i].shape, """The shape of tiffs and mask must be same, check your file name""";
                
        for i in self.labeled_mask.keys():
            self.labeled_mask[i] = np.expand_dims(rgb2LabeledMask(
                extractMaskByRGBs(self.labeled_mask[i], self.active_positive_class), self.active_positive_class), axis=2);
    
    def prepare_evaluated_mask(self, load_ids, evaluated_path): 
        
        self.load_evaluation(load_ids, evaluated_path);
        """Add evaluated buff to active_positive_class"""
        self.active_positive_class.append(self.plot.EvaluatedBuff);
        
        for i in self.evaluations.keys():
            self.evaluations[i] = np.expand_dims(rgb2LabeledMask(
                extractMaskByRGBs(self.evaluations[i], self.active_positive_class), self.active_positive_class), axis=2);
    
        
#####################################################
# Object-based image classification: super-pixel
#####################################################    
class Super_pixel_seg_dataset(TiffsDataset):    
    
    def __init__(self, tiff_path, label_path, plot, active_positive_class):
        TiffsDataset.__init__(self, tiff_path, label_path, plot, active_positive_class);
        
        self.centroids = {};
    
    def prepare_superpixel_dataset(self,
                                   is_train=True,
                                   load_ids=None,
                                   n_segments=10,
                                   batch_size=32,
                                   is_augment=True,
                                   rotate_clip=True, 
                                   random_histogram_eq=0.2, 
                                   random_brightness=(0.5, 2.0), 
                                   random_intensity=0.3,
                                   random_flip=0.75,
                                   model_input_pixel_size=(64, 64),
                                   one_hot=True,
                                   save_segments=False,
                                   evaluated_path=None,
                                   exclude_boundary_objs=False,
                                   boundary_width=1):
        
        assert load_ids is not None, "load_ids is None.";
        assert n_segments > 0, "n_segments must greater than 0.";
        assert random_histogram_eq >= 0 and random_histogram_eq <= 1;
        assert random_brightness[0] >= 0 and random_brightness[1] >= random_brightness[0];
        assert random_intensity >= 0 and random_intensity <= 1;
        assert random_flip >= 0 and random_flip <= 1;
        
        self.is_train = is_train;
        self.batch_size = batch_size;
        self.is_augment = is_augment;"""If is_augment is set to False, the other data augmentation parameters is invalid"""
        self.rotate_clip = rotate_clip; 
        self.random_histogram_eq = random_histogram_eq; 
        self.random_brightness = random_brightness;
        self.random_intensity = random_intensity;
        self.random_flip = random_flip;
        self.model_input_pixel_size = model_input_pixel_size;
        self.one_hot = one_hot;
        self.save_segments = save_segments;
        self.current = 0;
        
        if is_train and not is_augment:
            print("Warning: Training without data augmentation.");
        if not is_train and is_augment:
            print("Warning: Predicting using data augmentation.");
        
        self.centroids = {};
        self.prepare_dataset(load_ids);
        if not is_train:
            assert evaluated_path is not None;
            self.prepare_evaluated_mask(load_ids, evaluated_path)
        
        tiff_areas = np.array([i.shape[0] * i.shape[1] for i in self.tiffs.values()]);
        each_tiff_segmented_count = tiff_areas / 10000. * n_segments;
        each_tiff_segmented_count = np.round(each_tiff_segmented_count);
        each_tiff_segmented_count = dict(zip(self.tiffs.keys(), each_tiff_segmented_count));
        
        self.segments = {};
        print("The segmentation (super pixel) count of each trained tiff image: \n", each_tiff_segmented_count);
        
        """segmented by slic"""
        from skimage.segmentation import slic;
        from skimage.measure import regionprops;
        
        for i in load_ids:
            print("Tiff", i, " is being segmented ...");
            """Segmentation is time-consuming."""
            segments = slic(self.tiffs[i], n_segments=int(each_tiff_segmented_count[i]), slic_zero=True);
            
            segments = segments + 1; 

            """Some segmentation algorithms have the same segmented result each time, 
            but others can not, so we save it."""
            if self.save_segments:
                self.segments[i] = segments;
            
            objs = regionprops(segments);
            print("Tiff", i, " is divided into ", len(objs), " segments");
            
            centroids_row = np.array([j.centroid[0] for j in objs]);
            centroids_col = np.array([j.centroid[1] for j in objs]);
            centroids_row = np.round(centroids_row).astype(np.int32);
            centroids_col = np.round(centroids_col).astype(np.int32);
            
            if self.is_train:
                centroids_class = self.labeled_mask[i][centroids_row, centroids_col, 0].astype(np.int32);
            else:
                """load_evaluated is true which means the class contain the class of evaluated buff."""
                centroids_class = self.evaluations[i][centroids_row, centroids_col, 0].astype(np.int32);
            
            pic_id = np.array([i] * centroids_row.shape[0]);
            fids = np.array([int(j.label) for j in objs]);
            
            """Fids and pic_id can determine a object."""
            self.centroids[i] = np.array(list(zip(pic_id, centroids_row, centroids_col, centroids_class, fids)));
        
        self.centroids = np.concatenate(list(self.centroids.values()));
        
        print("Count of current centroids is ", self.centroids.shape[0]);
        if exclude_boundary_objs:
            self.exclude_Class_Boundary_Point(boundary_width=boundary_width);
            print("After excluding boundary points, the count of current centroids is ", self.centroids.shape[0]);
        
        if self.is_train:
            print("Train datasest is avialiable.");
        else:
            print("Test datasest is avialiable.");
        
        """In object-based image classification, labeled masks are no longer needed"""
        del self.labeled_mask;
        
    def next(self):
        
        batch_x, batch_y, pic_ids, obj_fids = [], [], [], [];
        
        #mask_x = [];
        """Shuffle the data after each complete pass"""
        if self.current >= self.centroids.shape[0]:
            self.current = 0;
            if self.is_train:
                np.random.shuffle(self.centroids);
        
        cur_centroids = self.centroids[self.current:self.current+self.batch_size, :];
        cur_tif_ids = np.unique(cur_centroids[:, 0]);
        if self.rotate_clip and self.is_augment:
            for i in cur_tif_ids:
                x, _ = _random_rotate_clip_image(self.tiffs[i], cur_centroids[cur_centroids[:, 0] == i, 1:3], \
                                                 self.model_input_pixel_size);
                batch_x.extend(x);
                batch_y.extend(cur_centroids[cur_centroids[:, 0] == i, -2]);
                if not self.is_train:
                    pic_ids.extend(cur_centroids[cur_centroids[:, 0] == i, 0]);
                    obj_fids.extend(cur_centroids[cur_centroids[:, 0] == i, -1]);
                #mask_x.extend(_random_rotate_clip_image(self.labeled_mask[i], cur_centroids[cur_centroids[:, 0] == i, 1:3], model_input_pixel_size, rotate_seed=seed)[0]);
        else:
            for i in cur_tif_ids:
                batch_x.extend(_clip_image(self.tiffs[i], cur_centroids[cur_centroids[:, 0] == i, 1:3], \
                                           self.model_input_pixel_size));
                batch_y.extend(cur_centroids[cur_centroids[:, 0] == i, -2]);
                if not self.is_train:
                    pic_ids.extend(cur_centroids[cur_centroids[:, 0] == i, 0]);
                    obj_fids.extend(cur_centroids[cur_centroids[:, 0] == i, -1]);
                #mask_x.extend(_clip_image(self.labeled_mask[i], cur_centroids[cur_centroids[:, 0] == i, 1:3], model_input_pixel_size));
                
        batch_x = np.array(batch_x);
        batch_y = np.array(batch_y);
        #mask_x = np.array(mask_x);
        
        if self.is_train:
            tmp_inds = np.arange(batch_x.shape[0]);
            np.random.shuffle(tmp_inds);
            batch_x = batch_x[tmp_inds, :];
            batch_y = batch_y[tmp_inds];
        else:
            pic_ids = np.array(pic_ids);
            obj_fids = np.array(obj_fids);
        #mask_x = mask_x[tmp_inds, :];
        
        cur_batch_size = batch_x.shape[0];
        """Data augment"""
        assert batch_x.shape[1] == self.model_input_pixel_size[0];
        assert batch_x.shape[2] == self.model_input_pixel_size[1];
        
        if self.is_augment:
            """histogram equal"""
            batch_x = batch_operation([batch_x, [self.random_histogram_eq]*cur_batch_size], \
                                      opea_func=_random_histogram_eq_img);
            
            """random brightnes"""
            batch_x = batch_operation([batch_x, [self.random_brightness[0]]*cur_batch_size, \
                                       [self.random_brightness[1]]*cur_batch_size], opea_func=_random_brightness_img);
            
            """random intensity"""
            batch_x = batch_operation([batch_x, [self.random_intensity]*cur_batch_size], \
                                      opea_func=_random_intensity_img);
            
            """random flip"""
            temp = batch_operation([batch_x, [self.random_flip]*cur_batch_size], opea_func=_random_flip_img);
            batch_x = np.array(list(temp[:, 0]));
        
        #flip_modes = temp[:, 1];
        #mask_x = batch_operation([mask_x, [random_flip]*cur_batch_size, flip_modes], opea_func=_random_flip_img)[:, 0];
        #mask_x = mask2BinaryMask(np.array(list(mask_x)));
        #plt.subplot(121);plt.imshow(batch_x[id]);plt.subplot(122);plt.imshow(mask_x[id][:, :, 0]);plt.show();
        
        if self.one_hot:
            batch_y = to_categorical(batch_y, self.classes);
        
        self.current = self.current + cur_batch_size;
        
        if self.is_train:
            return batch_x, batch_y;
        else:
            return batch_x, batch_y, pic_ids, obj_fids;
    
    def exclude_Class_Boundary_Point(self, boundary_width=1):
        from skimage.measure import regionprops;
        from skimage.segmentation import find_boundaries;
        print("===============Excluding the class boundary points.===============")
        
        for i in self.load_ids:
            curMask = self.labeled_mask[i].astype(np.int32);
            class_objs = regionprops(curMask);
            
            boundary_points = [];
            for j in range(len(class_objs)):
                curClasMask = np.copy(curMask[:, :, 0]);
                curClasMask[curClasMask != class_objs[j].label] = 0;
                
                bp = find_boundaries(curClasMask, mode='thick', background=self.plot.Background.value[1]);
                bp = np.where(bp==True);
                bp = list(zip(bp[0], bp[1]));
                boundary_points.extend(bp);
            
            boundary_points = np.unique(np.array(boundary_points).astype(np.int32), axis=0);
            
            boundary_points_u = boundary_points - (boundary_width-1, 0);
            boundary_points_b = boundary_points + (boundary_width-1, 0);
            boundary_points_l = boundary_points - (0, boundary_width-1);
            boundary_points_r = boundary_points + (0, boundary_width-1);
            
            boundary_points = np.concatenate((boundary_points, 
                                              boundary_points_u,
                                              boundary_points_b,
                                              boundary_points_l,
                                              boundary_points_r), axis=0);
            del boundary_points_u; del boundary_points_b;
            del boundary_points_l; del boundary_points_r;
            boundary_points = np.unique(boundary_points, axis=0);
            
            rs = curMask.shape[0];
            cs = curMask.shape[1];
            
            boundary_points = boundary_points[boundary_points[:, 0] >= 0];
            boundary_points = boundary_points[boundary_points[:, 0] < rs]; 
            boundary_points = boundary_points[boundary_points[:, 1] >= 0];
            boundary_points = boundary_points[boundary_points[:, 1] < cs]; 
            
            boundary_points_mask = np.zeros([rs, cs]);
            boundary_points_mask[boundary_points[:, 0], boundary_points[:, 1]] = 1;
            del boundary_points; del curMask;
            
            cur_pic_centroids = self.centroids[self.centroids[:, 0] == i];
            cur_pic_centroids = cur_pic_centroids[boundary_points_mask[cur_pic_centroids[:, 1], cur_pic_centroids[:, 2]] != 1, :];
            print("After excluding class boundary points, the pic ", i, " remains ", cur_pic_centroids.shape[0], " samples");
            self.centroids = np.concatenate((cur_pic_centroids, self.centroids[self.centroids[:, 0] != i]));
            del cur_pic_centroids; del boundary_points_mask;
            
    def get_n_samples(self):
        return self.centroids.shape[0];
    
    def get_class_weight(self):
        classes = np.unique(self.centroids[:, -2]);
        classes.sort();
        return compute_class_weight('balanced', classes, self.centroids[:, -2]), classes;


#####################################################
# Patch-based image classification: fcn
#####################################################    
class Patch_based_dataset(TiffsDataset):  
    def __init__(self, tiff_path, label_path, plot, active_positive_class):
        TiffsDataset.__init__(self, tiff_path, label_path, plot, active_positive_class);
        
        """Using a random clip to obtain training dataset."""
        self.centroids = {};
        self.random_range_each_tiff = {};
    
    def prepare_patch_based_dataset(self,
                                    is_train=True,
                                    load_ids=None,
                                    batch_size=32,
                                    class_mode="categorical",
                                    classes=2,
                                    is_augment=True,
                                    rotate_clip=True, 
                                    random_histogram_eq=0.2, 
                                    random_brightness=(0.5, 2.0), 
                                    random_intensity=0.3,
                                    random_flip=0.75,
                                    model_input_pixel_size=(256, 256),
                                    predict_center_pixel_size=None,
                                    evaluated_path=None):
    
        assert load_ids is not None, "load_ids is None.";
        assert class_mode in ["binary", "categorical"];
        assert random_histogram_eq >= 0 and random_histogram_eq <= 1;
        assert random_brightness[0] >= 0 and random_brightness[1] >= random_brightness[0];
        assert random_intensity >= 0 and random_intensity <= 1;
        assert random_flip >= 0 and random_flip <= 1;
        
        self.is_train = is_train;
        self.batch_size = batch_size;
        self.class_mode = class_mode;
        self.classes = classes;
        self.is_augment = is_augment;"""If is_augment is set to False, the other data augmentation parameters is invalid"""
        self.rotate_clip = rotate_clip; 
        self.random_histogram_eq = random_histogram_eq; 
        self.random_brightness = random_brightness;
        self.random_intensity = random_intensity;
        self.random_flip = random_flip;
        self.model_input_pixel_size = model_input_pixel_size;
        self.predict_center_pixel_size =  predict_center_pixel_size;
        
        if is_train and not is_augment:
            print("Warning: Training without data augmentation.");
        if not is_train and is_augment:
            print("Warning: Predicting using data augmentation.");
        
        self.centroids = {};
        self.random_range_each_tiff = {};
        self.prepare_dataset(load_ids);
        
        if not is_train:
            assert evaluated_path is not None;
            self.prepare_evaluated_mask(load_ids, evaluated_path)
        
        for i in load_ids:
            
            cur_tif_row = self.tiffs[i].shape[0];
            cur_tif_col = self.tiffs[i].shape[1];
            
            """The default random range is the entire tiff image."""
            self.random_range_each_tiff[i] = (cur_tif_row, cur_tif_col);
        
            if self.class_mode == "categorical" and is_train:
                self.labeled_mask[i] = \
                    to_categorical(self.labeled_mask[i], self.classes).reshape((cur_tif_row, cur_tif_col, self.classes));
        
        """The default output pixel size is the half of input pixel size.
            output_center_pixel_size was only used for generating test dataset.
        """
        if self.is_train:
            self.pool = multiprocessing.pool.ThreadPool(processes=min(self.batch_size, 32));
        else:
            if predict_center_pixel_size is None:
                predict_center_pixel_size = (int(self.model_input_pixel_size[0]/2.), 
                                             int(self.model_input_pixel_size[1]/2.));
            
            self.centroids = gen_pbic_test_data(load_ids, self.random_range_each_tiff, self.predict_center_pixel_size);   
            self.current = 0;
            
        if self.is_train:
            print("Train datasest is avialiable.");
        else:
            print("Test datasest is avialiable.");
    
    def _next_rand_batch(self, n_need):
        batch_x, batch_y = [], [];
        "Generating training data."
        rand_pic_ids = list(self.load_ids) * n_need;
        np.random.shuffle(rand_pic_ids);
        rand_pic_ids = rand_pic_ids[:n_need];
        rand_rows = [];
        rand_cols = [];
        for i in rand_pic_ids:
            rs = self.random_range_each_tiff[i][0];
            cs = self.random_range_each_tiff[i][1];
            
            rand_rows.append(np.random.randint(0, rs));
            rand_cols.append(np.random.randint(0, cs));
            
        rand_centroids = np.array(list(zip(rand_pic_ids, rand_rows, rand_cols)));
        
        cur_tif_ids = np.unique(rand_centroids[:, 0]);
        if self.rotate_clip and self.is_augment:
            for i in cur_tif_ids:
                x, x_seed = _random_rotate_clip_image(self.tiffs[i], rand_centroids[rand_centroids[:, 0] == i, 1:3], self.model_input_pixel_size);
                batch_x.extend(x);
                y, _ = _random_rotate_clip_image(self.labeled_mask[i], rand_centroids[rand_centroids[:, 0] == i, 1:3], \
                                                 self.model_input_pixel_size, rotate_seed=x_seed);
                batch_y.extend(y);
        else:
            for i in cur_tif_ids:
                batch_x.extend(_clip_image(self.tiffs[i], rand_centroids[rand_centroids[:, 0] == i, 1:3], self.model_input_pixel_size));
                batch_y.extend(_clip_image(self.labeled_mask[i], rand_centroids[rand_centroids[:, 0] == i, 1:3], self.model_input_pixel_size));
        
        batch_x = np.array(batch_x);
        batch_y = np.array(batch_y);
        
        tmp_inds = np.arange(batch_x.shape[0]);
        np.random.shuffle(tmp_inds);
        batch_x = batch_x[tmp_inds, :];
        batch_y = batch_y[tmp_inds, :];
        
        cur_batch_size = batch_x.shape[0];
        """Data augment"""
        assert batch_x.shape[1] == self.model_input_pixel_size[0];
        assert batch_x.shape[2] == self.model_input_pixel_size[1];
        
        if self.is_augment:
            """histogram equal"""
            batch_x = batch_operation([batch_x, [self.random_histogram_eq]*cur_batch_size], opea_func=_random_histogram_eq_img);
            """random brightnes"""
            batch_x = batch_operation([batch_x, [self.random_brightness[0]]*cur_batch_size, [self.random_brightness[1]]*cur_batch_size], \
                                      opea_func=_random_brightness_img);
            """random intensity"""
            batch_x = batch_operation([batch_x, [self.random_intensity]*cur_batch_size], opea_func=_random_intensity_img);
            """random flip"""
            bx = batch_operation([batch_x, [self.random_flip]*cur_batch_size], opea_func=_random_flip_img);
            batch_x = np.array(list(bx[:, 0]));
            batch_y = batch_operation([batch_y, [self.random_flip]*cur_batch_size, bx[:, 1]], opea_func=_random_flip_img)[:, 0];
        batch_y = np.round(np.array(list(batch_y)));
        return batch_x, batch_y;
    
    def next(self):
        
        batch_x, batch_y, pic_ids = [], [], [];
        
        if self.is_train:
            results = self.pool.map_async(self._next_rand_batch, [1]*self.batch_size);
            results = results.get();
            batch_x, batch_y = list(zip(*results));
            batch_x = np.concatenate(batch_x);
            batch_y = np.concatenate(batch_y);
            return batch_x, batch_y;
        else:
            assert self.centroids.shape != 0;
            pic_ids = [];
            centros = [];
            #loc_ids = [];
            
            if self.current >= self.centroids.shape[0]:
                self.current = 0;
            
            cur_centroids = self.centroids[self.current:self.current+self.batch_size, :];
            cur_tif_ids = np.unique(cur_centroids[:, 0]);
            for i in cur_tif_ids:
                batch_x.extend(_clip_image(self.tiffs[i], cur_centroids[cur_centroids[:, 0] == i, 1:3], self.model_input_pixel_size));
                pic_ids.extend(cur_centroids[cur_centroids[:, 0] == i, 0]);
                centros.extend(cur_centroids[cur_centroids[:, 0] == i, 1:3]);
                #loc_ids.extend(cur_centroids[cur_centroids[:, 0] == i, -1]);
                #batch_y.extend(_clip_image(self.labeled_mask[i], cur_centroids[cur_centroids[:, 0] == i, 1:3], self.model_input_pixel_size));
                    
            batch_x = np.array(batch_x);
            pic_ids = np.array(pic_ids);
            centros = np.array(centros);
            #batch_y = np.array(batch_y);
            #loc_ids = np.array(loc_ids);
            
            cur_batch_size = batch_x.shape[0];
            
            """Data augment"""
            assert batch_x.shape[1] == self.model_input_pixel_size[0];
            assert batch_x.shape[2] == self.model_input_pixel_size[1];
            
            if self.is_augment:
                """histogram equal"""
                batch_x = batch_operation([batch_x, [self.random_histogram_eq]*cur_batch_size], opea_func=_random_histogram_eq_img);
                """random brightnes"""
                batch_x = batch_operation([batch_x, [self.random_brightness[0]]*cur_batch_size, [self.random_brightness[1]]*cur_batch_size], opea_func=_random_brightness_img);
                """random intensity"""
                batch_x = batch_operation([batch_x, [self.random_intensity]*cur_batch_size], opea_func=_random_intensity_img);
                
            self.current = self.current + cur_batch_size;
            #return batch_x, pic_ids, loc_ids, centros;
            return batch_x, pic_ids, centros;
    
    def get_n_samples(self):
        if self.is_train:
            return 1024;
        else:
            return self.centroids.shape[0];


#####################################################
# Vaihingen dataset
#####################################################
class Vaihingen_class(Enum):
    Building = [(0, 0, 255), 1, 'Building'];
    Tree = [(0, 255, 0), 2, 'Tree'];
    Car = [(255, 255, 0), 3, 'Car'];
    Low_vegetation = [(0, 255, 255), 4, 'Low_vegetation'];
    Background = [(255, 255, 255), 0, 'Background'];
    
    """EvaluatedBuff only exists in evaluated mask."""
    EvaluatedBuff = [(0, 0, 0), 5, 'EvaluatedBuff'];














