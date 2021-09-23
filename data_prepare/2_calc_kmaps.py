import numpy as np
import os
import sys;
import skimage.io as io
import glob
sys.path.append("..")
from spatial_analysis_utils_v2_sh import *

# Calculates the cross K-function at each cell and propagate the same values to all the pixels in the cell connected components in the ground truth dilated dot map or binary mask.

# Configuration Variables

# Configure data input/output paths
image_dir='../../datasets/lungdata_cc/images'
gt_dir='../../datasets/lungdata_cc/gt_custom_all'
out_dir = '../../datasets/lungdata_cc/k_func_maps'

# Configure K function
do_k_correction=True
n_classes = 3
r_step = 15 # r stands for radius
r_range = range(0, 100, r_step)
r_list = [*r_range]
r_classes = len(r_range)
r_classes_all = r_classes * (n_classes)

if __name__ == "__main__":
    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    img_path_list = glob.glob(os.path.join(image_dir, '*.png'))

    # Calculate the cross K-function at each cell and propagate the same values to all the pixels in the cell connected components in the ground truth dilated dot map or binary mask.
    for img_path in img_path_list:
        # Load the ground truth dot maps and gaussian maps or binary masks
        print('img', img_path )
        img_name = os.path.basename(img_path)
        gt_path = os.path.join(gt_dir,img_name.replace('.png','_gt_dots.npy'));
        gt_dots=np.load(gt_path, allow_pickle=True)[:,:,1:].squeeze()
        gt_dmap_path = os.path.join(gt_dir,img_name.replace('.png','.npy'));
        gt_dmap=np.load(gt_dmap_path, allow_pickle=True)[:,:,1:].squeeze()
        gt_dots_all = gt_dots.max(-1)
        gt_dmap = gt_dmap > 0
        gt_dmap_all = gt_dmap.max(-1)
        gt_dmap_all_comp = label(gt_dmap_all) # label all connected components to make it easy to propagate the k function to the cell's pixels
        gt_kmap_out_path = os.path.join(out_dir,img_name.replace('.png','_gt_kmap.npy')); # output filepath
        k_area = gt_dots.shape[0]*gt_dots.shape[1]

        # cells_y, cells_x arrays to hold cell dot coordinates, cells_mark array to hold cells classes index
        # First entry is reserved to be used as the center cell, which will have its own identifier, when computing the K-function, and so it will change with center cell changing
        cells_y=[0]
        cells_x=[0]
        cells_mark=['1000']

        # load the cells' coordinates and classes into cells_y, cells_x, and cells_mark
        for c in range(n_classes):
            c_points = np.where(gt_dots[:,:, c] > 0)
            if(len(c_points[0])>0):
                cells_y = np.concatenate((cells_y, c_points[0]))
                cells_x = np.concatenate((cells_x, c_points[1]))
                cells_mark =  cells_mark + [str(c+1)]*len(c_points[0])

        # initialize the kmap array
        gt_kmap = np.zeros((gt_dots.shape[0],gt_dots.shape[1],r_classes_all))

        # c_points are the center points, which are all the cells
        c_points = np.where(gt_dots_all > 0)
        if(len(c_points[0]) == 0):
            continue

        '''
            Loop over each cell in c_points
                Set the first entry in cells_y and cells_x to the current cell coordinates
                Compute the K-function with respect to each of the other classes.
                In the kmap, set all the pixels in the current cell connected component to the calculated k-function
        '''
        for ci in range(len(c_points[0])):
            cy = c_points[0][ci]
            cx = c_points[1][ci]
            comp_indx = gt_dmap_all_comp[cy,cx]
            cells_y[0] = cy
            cells_x[0] = cx
            cells_ppp = ppp(cells_x, cells_y, cells_mark) # the point pattern in R object
            k_indx = 0
            c_k_func = np.zeros(r_classes_all)
            for s2 in range(n_classes):
                if(gt_dots[:,:, s2].sum() > 0):
                    if(do_k_correction):
                        r_Kcross, K_val_samp = Kcross(cells_ppp, i='1000', j=str(s2+1), correction='iso', plot=False, r=r_range)
                    else:
                        r_Kcross, K_val_samp = Kcross(cells_ppp, i='1000', j=str(s2+1), correction='none', plot=False, r=r_range)
                    c_k_func[k_indx:k_indx+r_classes] = K_val_samp/k_area * gt_dots[:,:, s2].sum()
                else:
                    c_k_func[k_indx:k_indx+r_classes] = 0
                k_indx += r_classes
            gt_kmap[gt_dmap_all_comp == comp_indx] = c_k_func
        gt_kmap.dump(gt_kmap_out_path)

