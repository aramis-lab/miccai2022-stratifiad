#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:34:22 2022
    Sorbonne Université
    Paris Brain Institute (INSERM, CNRS, Sorbonne Univeristé, AP-HP), INRIA "ARAMIS Lab"
@author: mehdi.ounissi
@email : mehdi.ounissi@icm-institue.org
         mehdi.ounissi@etu.sorbonne-universite.fr
"""
from skimage.exposure import match_histograms
from torch.utils.data import Dataset
from tqdm import trange
from PIL import Image
import numpy as np
import logging
import torch


def dice_coeff_batch(batch_bn_mask, batch_true_bn_mask):
    """ dice_coeff_batch : function that returns the mean dice coeff for a batch of pairs 
    mask, ground truth mask """
    
    def single_dice_coeff(input_bn_mask, true_bn_mask):
        """single_dice_coeff : function that returns the dice coeff for one pair 
        of mask and ground truth mask"""

        # The eps value is used for numerical stability
        eps = 0.0001

        # Computing intersection and union masks
        inter_mask = torch.dot(input_bn_mask.view(-1), true_bn_mask.view(-1))
        union_mask = torch.sum(input_bn_mask) + torch.sum(true_bn_mask) + eps

        # Computing the Dice coefficient
        return (2 * inter_mask.float() + eps) / union_mask.float()

    # Init dice score for batch (GPU ready)
    if batch_bn_mask.is_cuda: dice_score = torch.FloatTensor(1).cuda().zero_()
    else: dice_score = torch.FloatTensor(1).zero_()

    # Compute Dice coefficient for the given batch
    for pair_idx, inputs in enumerate(zip(batch_bn_mask, batch_true_bn_mask)):
        dice_score +=  single_dice_coeff(inputs[0], inputs[1])
    
    # Return the mean Dice coefficient over the given batch
    return dice_score / (pair_idx + 1)

def DiceLoss(y_hat, y):
    y_hat = torch.sigmoid(y_hat) 
    dice_loss = 1 - dice_coeff_batch(y_hat, y)
    return dice_loss

def metrics(p_n, tp, fp, tn, fn):
    """ Returns accuracy, precision, recall, f1 based on the inputs 
    tp : true positives, fp: false positives, tn: true negatives, fn: false negatives
    For details please check : https://en.wikipedia.org/wiki/Precision_and_recall
    """
    try:
        # Computing the accuracy
        accuracy  = (tp + tn) / p_n

        # Computing the precision
        precision =  tp / (tp + fp)

        # Computing the recall
        recall    =  tp / (tp + fn)

        # Computing the f1
        f1        =  2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError:
        precision, recall, accuracy, f1 = 0, 0, 0, 0
        
    return precision, recall, accuracy, f1

def confusion_matrix(prediction, truth):
    """ Returns the confusion matrix for the values in the `prediction` and `truth`
    tensors, i.e. the amount of positions where the values of `prediction`
    and `truth` are
    - 1 and 1 (True Positive)
    - 1 and 0 (False Positive)
    - 0 and 0 (True Negative)
    - 0 and 1 (False Negative)
    # Source of the confusion_matrix function: https://gist.github.com/the-bass
    """
    # Computing the confusion vector
    confusion_vector = prediction / truth
    
    # Element-wise division of the 2 tensors returns a new tensor which holds a
    # unique value for each case:
    #   1     where prediction and truth are 1 (True Positive)
    #   inf   where prediction is 1 and truth is 0 (False Positive)
    #   nan   where prediction and truth are 0 (True Negative)
    #   0     where prediction is 0 and truth is 1 (False Negative)
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()

    # Computing the total (p+n)
    p_n = tp + fp + tn + fn

    # Computing the precision, recall, accuracy, f1 metrics
    precision, recall, accuracy, f1 = metrics(p_n, tp, fp, tn, fn)

    return tp/p_n, fp/p_n, tn/p_n, fn/p_n, precision, recall, accuracy, f1 



class CustomDataset(Dataset):
    """ CustomDataset : Class that loads data (images and masks) in efficent way"""
    def __init__(self, imgs_dirs, masks_dirs, ref_image_path, normalize=False,cached_data=True, n_channels=1,scale=1):
        self.imgs_dirs = imgs_dirs    # All paths to images 
        self.masks_dirs = masks_dirs  # All paths to masks 
        self.scale = scale            # image and mask scale
        self.n_channels = n_channels  # input model channels
        self.normalize = normalize    # normalization switch

        # Make sure the scale is between [0, 1]
        assert 0 < scale <= 1, '[ERROR] Scale must be between 0 and 1'

        # Load the reference image into RAM
        ref_image = Image.open(ref_image_path)
        
        if np.array(ref_image).shape[-1] > n_channels:
            r, g, b, a = ref_image.split()
            ref_image = Image.merge("RGB", (r,g,b))
        
        # Save the reference image into RAM to be used
        self.ref_image = ref_image.copy()

        # Caching the dataset (WARRING : this needs to be used when you have big RAM memory)
        if cached_data:
            logging.info(f'[INFO] Caching the given dataset with {len(self.imgs_dirs)} images and {len(self.masks_dirs)} masks')
            # Turn on the cach flag
            self.cached_dataset = True

            # Preparing the images and masks lists
            self.cache_imgs, self.cache_masks = [], []
            
            # Cache & pre-process the images and the masks (train/val) ready
            for i in trange(len(imgs_dirs)):
                pil_img = Image.open(self.imgs_dirs[i])
                np_img = self.preprocess(pil_img, self.ref_image, self.n_channels, self.scale, self.normalize, mask=False)
                self.cache_imgs.append(np_img)

                if len(self.masks_dirs) == len(self.imgs_dirs):
                    pil_mask = Image.open(self.masks_dirs[i])
                    np_img = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)
                    self.cache_masks.append(np_img)
        else:
            logging.info(f'[INFO] Dataset with {len(self.imgs_dirs)} images and {len(self.masks_dirs)} masks')
            
    def __len__(self): return len(self.imgs_dirs)

    def delete_cached_dataset(self):
        try:
            del self.cache_imgs[:]
            del self.cache_masks[:]
            logging.info(f'[INFO] All cache deleted')
            return True
        except:
            return False

    def preprocess(self, pil_img, ref_image, n_channels, scale, normalize, mask=True):
        if not(mask):

            if np.array(pil_img).shape[-1] > n_channels:
                r, g, b, a = pil_img.split()
                pil_img = Image.merge("RGB", (r,g,b))

             # This part is concerns the normalization 
            if normalize:
                # Make sure the reference image and the current image have the same size
                assert pil_img.size == ref_image.size, \
                f'Image and reference image should be the same size for histograms matching, but are {pil_img.size} and {ref_image.size}'
           
                if n_channels == 3: pil_img = Image.fromarray(match_histograms(np.array(pil_img),np.array(ref_image), multichannel=True))
                else: pil_img = Image.fromarray(match_histograms(np.array(pil_img),np.array(ref_image)))
            
        # Rescale the image if needed
        if scale != 1 :
            # Get the H and W of the img
            w, h = pil_img.size

            # Get the extimated new size
            newW, newH = int(scale * w), int(scale * h)

            # Resize the image according the given scale
            pil_img = pil_img.resize((newW, newH))

        # Uncomment to convert imgs into gray scale imgs
        # if n_channels != 3: pil_img = pil_img.convert("L")

        # Convert the PIL image into numpy array
        np_img = np.array(pil_img)

        # Add an extra dim if only H, W image
        if len(np_img.shape) == 2: np_img = np.expand_dims(np_img, axis=2)

        # Re-arange the image from (H, W, C) to (C ,H ,W)
        np_img_ready = np_img.transpose((2, 0, 1))
        
        # Ensure the imgs to be in [0, 1]
        if np_img_ready.max() > 1: np_img_ready = np_img_ready / 255
        
        return np_img_ready
    
    def __getitem__(self, i):
        # When the dataset is cached load the img and mask from RAM
        if self.cached_dataset:
            np_img = self.cache_imgs[i]
            if len(self.masks_dirs) == len(self.imgs_dirs):
                np_mask = self.cache_masks[i]
        
        # Otherwise load the img and mask from Disk to RAM
        else:
            # Load the image 
            img_dir = self.imgs_dirs[i]
            pil_img = Image.open(img_dir)

            # Preprocess the image 
            np_img = self.preprocess(pil_img, self.ref_image, self.n_channels, self.scale, self.normalize, mask=False)

            # Load & pre-process the mask if possible
            if len(self.masks_dirs) == len(self.imgs_dirs):
                mask_dir = self.masks_dirs[i]
                pil_mask = Image.open(mask_dir)
                np_mask = self.preprocess(pil_mask, self.ref_image, self.n_channels, self.scale, self.normalize, mask=True)
            
            
        if len(self.masks_dirs) == len(self.imgs_dirs):
            return {
                'image': torch.from_numpy(np_img).type(torch.FloatTensor),
                'mask': torch.from_numpy(np_mask).type(torch.FloatTensor)
            }
        else:
            return {
                'image': torch.from_numpy(np_img).type(torch.FloatTensor)
            }

