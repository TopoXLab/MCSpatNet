from torch.utils.data import Dataset
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from torchvision import transforms
import random
from PIL import Image
import glob
import skimage.io as io


class CellsDataset(Dataset):
    def __init__(self,img_root, gt_dmap_root, gt_dots_root, class_indx, split_filepath=None, phase='train', fixed_size=-1, max_side=-1, max_scale=-1, return_padding=False):
        super(CellsDataset, self).__init__()
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth dilated dot maps.
        gt_dots_root: the root path of ground-truth dot maps.
        class_indx: a comma separated list of channel indices to return from the ground truth
        split_filepath: if not None, then use only the images in the file
        phase: train or test
        fixed_size:  if > 0 return crops of side=fixed size during training
        max_side: boolean indicates whether to have a maximum side length during training
        max_scale: apply padding to make the patch side divisible by max_scale
        return_padding: return the x and y padding added by max_scale
        '''
        self.img_root=img_root
        self.gt_dmap_root=gt_dmap_root
        self.gt_dots_root=gt_dots_root
        self.phase=phase
        self.return_padding = return_padding

        if(split_filepath is None):
            self.img_names=[filename for filename in os.listdir(img_root) \
                               if os.path.isfile(os.path.join(img_root,filename))]
        else:
            self.img_names=np.loadtxt(split_filepath, dtype=str).tolist()
            
        self.n_samples=len(self.img_names)

        self.fixed_size = fixed_size
        self.max_side = max_side
        self.max_scale = max_scale
        self.class_indx = class_indx
        self.class_indx_list = [int(x) for x in self.class_indx.split(',')]


    def __len__(self):
        return self.n_samples

    def __getitem__(self,index):
        assert index <= len(self), 'index range error'

        # Read image, normalize it, and make sure it is in RGB format
        img_name=self.img_names[index]
        print('img_name',img_name)
        img=io.imread(os.path.join(self.img_root,img_name))/255# convert from [0,255] to [0,1]
        if len(img.shape)==2: # expand grayscale image to three channel.
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),2)

        # Read ground truth dilated dot map
        gt_path = os.path.join(self.gt_dmap_root,img_name.replace('.png','.npy'));
        if(os.path.isfile(gt_path)):
            gt_dmap=np.load(gt_path, allow_pickle=True)[:,:,self.class_indx_list].squeeze()
        else:
            gt_dmap=np.zeros((img.shape[0], img.shape[1], len(self.class_indx_list)))

        # Read ground truth dot map
        gt_dots_path = os.path.join(self.gt_dots_root,img_name.replace('.png','_gt_dots.npy'));
        if(os.path.isfile(gt_dots_path)):
            gt_dots=np.load(gt_dots_path, allow_pickle=True)[:,:,self.class_indx_list].squeeze()
        else:
            gt_dots=np.zeros((img.shape[0], img.shape[1], len(self.class_indx_list)))

        
        # if train, apply random flipping augmentation
        if random.randint(0,1)==1 and self.phase=='train':
            img=img[:,::-1].copy() # horizontal flip
            gt_dmap=gt_dmap[:,::-1].copy() # horizontal flip
            gt_dots=gt_dots[:,::-1].copy() # horizontal flip
        
        if random.randint(0,1)==1 and self.phase=='train':
            img=img[::-1,:].copy() # vertical flip
            gt_dmap=gt_dmap[::-1,:].copy() # vertical flip
            gt_dots=gt_dots[::-1,:].copy() # vertical flip

        # if train, make sure width and height < max_side
        if(self.phase=='train' and self.max_side > 0):
            h = img.shape[0]
            w = img.shape[1]
            h2 = h
            w2 = w
            crop = False
            if(h > self.max_side):
                h2 = self.max_side
                crop = True
            if(w > self.max_side):
                w2 = self.max_side
                crop = True
            if(crop):
                y=0
                x=0
                if(not (h2 ==h)):
                    y = np.random.randint(0, high = h-h2)
                if(not (w2 ==w)):
                    x = np.random.randint(0, high = w-w2)
                img = img[y:y+h2, x:x+w2, :]
                gt_dmap = gt_dmap[y:y+h2, x:x+w2]
                gt_dots = gt_dots[y:y+h2, x:x+w2]

        
        # if train, make a random crop of size = fixed_size or if fixed_size<0 use 1/4 of image dimensions
        if self.phase=='train':
            i = -1
            img_pil = Image.fromarray(img.astype(np.uint8)*255);
            if(self.fixed_size < 0):
                i, j, h, w = transforms.RandomCrop.get_params(img_pil, output_size=(img.shape[0]//4, img.shape[1]//4))
            elif(self.fixed_size < img.shape[0] or self.fixed_size < img.shape[1]):
                i, j, h, w = transforms.RandomCrop.get_params(img_pil, output_size=(min(self.fixed_size,img.shape[0]), min(self.fixed_size,img.shape[1])))
            if(i >= 0):
                img = img[i:i+h, j:j+w, :]
                gt_dmap = gt_dmap[i:i+h, j:j+w]
                gt_dots = gt_dots[i:i+h, j:j+w]

        # Add padding to make sure image dimensions are divisible by max_scale
        pad_y1=0
        pad_y2=0
        pad_x1=0
        pad_x2=0
        if self.max_scale>1: # to downsample image and density-map to match deep-model.
            ds_rows=int(img.shape[0]//self.max_scale)*self.max_scale
            ds_cols=int(img.shape[1]//self.max_scale)*self.max_scale
            pad_y1 = 0
            pad_y2 = 0
            pad_x1 = 0
            pad_x2 = 0
            if(ds_rows < img.shape[0]):
                pad_y1 = (self.max_scale - (img.shape[0] - ds_rows))//2
                pad_y2 = (self.max_scale - (img.shape[0] - ds_rows)) - pad_y1
            if(ds_cols < img.shape[1]):
                pad_x1 = (self.max_scale - (img.shape[1] - ds_cols))//2
                pad_x2 = (self.max_scale - (img.shape[1] - ds_cols)) - pad_x1
            img = np.pad(img, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(1,) )# padding constant differs by dataset based on bg color
            gt_dmap = np.pad(gt_dmap, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(0,) )# padding constant differs by dataset based on bg color
            gt_dots = np.pad(gt_dots, ((pad_y1,pad_y2),(pad_x1,pad_x2),(0,0)), 'constant', constant_values=(0,) )# padding constant differs by dataset based on bg color


        # Covert image and ground truth to pytorch format
        img=img.transpose((2,0,1)) # convert to order (channel,rows,cols)
        if(len(self.class_indx_list) > 1):
            gt_dmap=gt_dmap.transpose((2,0,1)) # convert to order (channel,rows,cols)
            gt_dots=gt_dots.transpose((2,0,1)) # convert to order (channel,rows,cols)
        else:
            gt_dmap=gt_dmap[np.newaxis,:,:]
            gt_dots=gt_dots[np.newaxis,:,:]
        img_tensor=torch.tensor(img,dtype=torch.float)
        gt_dmap_tensor=torch.tensor(gt_dmap,dtype=torch.float)
        gt_dots_tensor=torch.tensor(gt_dots,dtype=torch.float)

        if(self.return_padding):
            return img_tensor,gt_dmap_tensor,gt_dots_tensor,img_name, (pad_y1, pad_y2, pad_x1, pad_x2)
        else:
            return img_tensor,gt_dmap_tensor,gt_dots_tensor,img_name


