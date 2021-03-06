import numpy as np
import glob
import os
import sys
import skimage.io as io
from scipy import ndimage
import scipy.io as sio
import cv2
import scipy

# Configuration variables
in_root_dir = '../../MCSpatNet_datasets/CoNSeP/Train'
out_root_dir = '../../MCSpatNet_datasets/CoNSeP_train'
# Define cell classes
classes_max_indx = 8
# lymph: blue,  tumor: red, stromal: green
color_set = {1: (0, 162, 232), 2: (255, 0, 0), 3: (0, 255, 0)}
# define mapping of cell classes in input to output
class_group_mapping_dict = {1:[2],2:[3,4],3:[1,5,6,7]}
n_grouped_class_channels = 4
# Define rescaling rate of images
img_scale = 0.5
remove_duplicates = False  # if True will remove duplicate of cells annotated within 5 pixel distance
'''
Original cell classes:
          1 = other 
	      2 = inflammatory 
	      3 = healthy epithelial 
	      4 = dysplastic/malignant epithelial 
          5 = fibroblast 
          6 = muscle  
	      7 = endothelial 
Grouped cell Classes: 
	      1 = inflammatory (sky blue)
	      2 = All epithelial (healthy epithelial + dysplastic/malignant epithelial) (red)
          3 = All stromal (fibroblast +  muscle  + endothelial + other) (green) 
'''

"""
    This code assumes the input has the following format:
        - Within <in_root_dir>: 
            Images folder: the image patches labelled from that slide 
            Labels folder: the mat files with the labels for each patch in images
        - The mat file has the following variables: 
            inst_centroid: array of shape n x 2, n is number of cells, and coordinates are (x,y)
            inst_type: array holding the cell class type for each cell in inst_centroid. The class types are sequential integers starting from 1. 
                        This is why the color_set dictionary has keys starting from 1 that represent the cell class types.

"""


def gaussian_filter_density(img, points, point_class_map, out_filepath, start_y=0, start_x=0, end_y=-1, end_x=-1):
    '''
        Build a KD-tree from the points, and for each point get the nearest neighbor point.
        The default Guassian width is 9.
        Sigma is adaptively selected to be min(nearest neighbor distance*0.125, 2) and truncate at 2*sigma.
        After generation of each point Gaussian, it is normalized and added to the final density map.
        A visualization of the generated maps is saved in <slide_name>_<img_name>.png and <slide_name>_<img_name>_binary.png
    '''
    img_shape = [img.shape[0], img.shape[1]]
    print("Shape of current image: ", img_shape, ". Totally need generate ", len(points), "gaussian kernels.")
    density = np.zeros(img_shape, dtype=np.float32)
    density_class = np.zeros((img.shape[0], img.shape[1], point_class_map.shape[2]), dtype=np.float32)
    if (end_y <= 0):
        end_y = img.shape[0]
    if (end_x <= 0):
        end_x = img.shape[1]
    gt_count = len(points)
    if gt_count == 0:
        return density
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(points.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(points, k=2)
    print('generate density...')

    max_sigma = 2;  # kernel size = 4, kernel_width=9

    for i, pt in enumerate(points):
        pt2d = np.zeros(img_shape, dtype=np.float32)
        if (pt[1] < start_y or pt[0] < start_x or pt[1] >= end_y or pt[0] >= end_x):
            continue
        pt[1] -= start_y
        pt[0] -= start_x
        if int(pt[1]) < img_shape[0] and int(pt[0]) < img_shape[1]:
            pt2d[int(pt[1]), int(pt[0])] = 1.
        else:
            continue
        if gt_count > 1:
            sigma = (distances[i][1]) * 0.125
            sigma = min(max_sigma, sigma)
        else:
            sigma = max_sigma;

        kernel_size = min(max_sigma * 2, int(2 * sigma + 0.5))
        sigma = kernel_size / 2
        kernel_width = kernel_size * 2 + 1
        # if(kernel_width < 9):
        #     print('i',i)
        #     print('distances',distances.shape)
        #     print('kernel_width',kernel_width)
        pnt_density = scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant', truncate=2)
        pnt_density /= pnt_density.sum()
        density += pnt_density
        class_indx = point_class_map[int(pt[1]), int(pt[0])].argmax()
        density_class[:, :, class_indx] = density_class[:, :, class_indx] + pnt_density

    #density_class.astype(np.float16).dump(out_filepath)
    #density.astype(np.float16).dump(os.path.splitext(out_filepath)[0] + '_all.npy')
    (density_class > 0).astype(np.uint8).dump(out_filepath)
    (density > 0).astype(np.uint8).dump(os.path.splitext(out_filepath)[0] + '_all.npy')
    #io.imsave(out_filepath.replace('.npy', '.png'), (density / density.max() * 255).astype(np.uint8))
    io.imsave(out_filepath.replace('.npy', '_binary.png'), ((density > 0) * 255).astype(np.uint8))
    for s in range(1, density_class.shape[-1]):
        io.imsave(out_filepath.replace('.npy', '_s' + str(s) + '_binary.png'),
                  ((density_class[:, :, s] > 0) * 255).astype(np.uint8))
    print('done.')
    return density.astype(np.float16), density_class.astype(np.float16)


if __name__ == "__main__":
    '''
        For each image: 
            Re-scale the patch image and the coordinates of the labelled cell centers. 
                Save rescaled image as (<out_img_dir>/<img_name>.png)
                and create a visualization of the cell classes with different colors overlaid on the patch image (saved as <out_gt_dir>/<img_name>_img_with_dots.jpg)
            Create classification dot annotation map (saved as <out_gt_dir>/<img_name>_gt_dots.npy) 
            Create detection dot annotation map ( saved as <out_gt_dir>/<img_name>_gt_dots_all.npy) 
            Generate binary mask, where a Gaussian is created at each cell center. The width of the Gaussian is adaptive such that cells do not intersect. 
            The Gaussian maps are saved as binary masks by setting all pixels > 0 to 1 and the rest to zero.
                Classification map file saved as <out_gt_dir>/<img_name>.npy 
                    and visualization of the binary masks saved as (<out_gt_dir>/<img_name>_s<class_indx>_binary.png )
                Detection map file saved as <out_gt_dir>/<img_name>_all.npy 
                    and visualization of the binary masks saved as (<out_gt_dir>/<img_name>_binary.png )

    '''
    '''
        Each .mat label file has the keys:
        'inst_type'
        'inst_centroid'
    '''
    in_img_dir = os.path.join(in_root_dir, 'Images')
    in_label_dir = os.path.join(in_root_dir, 'Labels')

    out_img_dir = os.path.join(out_root_dir, 'images')
    out_gt_dir = os.path.join(out_root_dir, 'gt_custom')
    if not os.path.exists(out_root_dir):
        os.mkdir(out_root_dir)
    if not os.path.exists(out_img_dir):
        os.mkdir(out_img_dir)
    if not os.path.exists(out_gt_dir):
        os.mkdir(out_gt_dir)

    img_files = glob.glob(os.path.join(in_img_dir, '*.png'))

    for img_filepath in img_files:
        print('img_filepath', img_filepath)

        # read img
        img_name = os.path.splitext(os.path.basename(img_filepath))[0]
        out_gt_dmap_filepath = os.path.join(out_gt_dir, img_name  + '.npy')
        img = io.imread(img_filepath)[:, :, 0:3]

        # read mat file
        mat_filepath = os.path.join(in_label_dir, img_name + '.mat')
        mat = sio.loadmat(mat_filepath)

        # read and scale centroids
        centroids = (mat["inst_centroid"] * img_scale).astype(int)
        class_types = mat["inst_type"].squeeze()

        # scale img
        img2 = cv2.resize(img, (int(img.shape[1] * img_scale + 0.5), int(img.shape[0] * img_scale + 0.5)))
        img3 = img2.copy()
        io.imsave(os.path.join(out_img_dir, img_name+'.png'), img2)

        # init label arr
        patch_label_arr_dots = np.zeros((img2.shape[0], img2.shape[1], classes_max_indx), dtype=np.uint8)

        # Make sure coordinates are within limits after scaling image
        # print('centroids',centroids.shape)
        # print('class_types',class_types.shape)
        centroids[(np.where(centroids[:, 1] >= img2.shape[0]), 1)] = img2.shape[0] - 1
        centroids[(np.where(centroids[:, 0] >= img2.shape[1]), 0)] = img2.shape[1] - 1

        # Generated of ground truth classification dot annotation map
        for dot_class in range(1, classes_max_indx):
            patch_label_arr = np.zeros((img2.shape[0], img2.shape[1]))
            patch_label_arr[(centroids[np.where(class_types == dot_class)][:, 1],
                                centroids[np.where(class_types == dot_class)][:, 0])] = 1
            patch_label_arr_dots[:, :, dot_class] = patch_label_arr
            #patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((5, 5)), mode='constant', cval=0.0)
            # img2[np.where(patch_label_arr > 0)] = color_set[dot_class]

        # Map indices in keys of  class_group_mapping_dict to all the classes in the item value
        patch_label_arr_dots_grouped = np.zeros((img2.shape[0], img2.shape[1], n_grouped_class_channels), dtype=np.uint8)
        for class_id, map_class_lst in class_group_mapping_dict.items():
            patch_label_arr = patch_label_arr_dots[:, :, map_class_lst].sum(axis=-1)
            patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((9, 9)), mode='constant', cval=0.0)
            img3[np.where(patch_label_arr > 0)] = color_set[class_id]
            patch_label_arr_dots_grouped[:, :, class_id] = patch_label_arr_dots[:, :, map_class_lst].sum(axis=-1)
        patch_label_arr_dots = patch_label_arr_dots_grouped

        # Remove duplicate dots
        if (remove_duplicates):
            for c in range(patch_label_arr_dots.shape[-1]):
                tmp = ndimage.convolve(patch_label_arr_dots[:, :, c], np.ones((5, 5)), mode='constant', cval=0.0)
                duplicate_points = np.where(tmp > 1)
                while (len(duplicate_points[0]) > 0):
                    y = duplicate_points[0][0]
                    x = duplicate_points[1][0]
                    patch_label_arr_dots[max(0, y - 2):min(patch_label_arr_dots.shape[0] - 1, y + 3),
                    max(0, x - 2):min(patch_label_arr_dots.shape[1] - 1, x + 3), c] = 0
                    patch_label_arr_dots[y, x, c] = 1
                    tmp = ndimage.convolve(patch_label_arr_dots[:, :, c], np.ones((5, 5)), mode='constant',
                                            cval=0.0)
                    duplicate_points = np.where(tmp > 1)

        # Generate the detection ground truth dot map
        patch_label_arr_dots_all = patch_label_arr_dots[:, :, :].sum(axis=-1)
        # Save Dot maps
        patch_label_arr_dots.astype(np.uint8).dump(
            os.path.join(out_gt_dir, img_name + '_gt_dots.npy'))
        patch_label_arr_dots_all.astype(np.uint8).dump(
            os.path.join(out_gt_dir, img_name + '_gt_dots_all.npy'))

        # Visualize of all cell types overlaid dots on img
        for dot_class in range(1, patch_label_arr_dots.shape[-1]):
            print('dot_class', dot_class)
            print('patch_label_arr_dots[:,:,dot_class]', patch_label_arr_dots[:, :, dot_class].sum())
            patch_label_arr = patch_label_arr_dots[:, :, dot_class].astype(int)
            patch_label_arr = ndimage.convolve(patch_label_arr, np.ones((5, 5)), mode='constant', cval=0.0)
            img2[np.where(patch_label_arr > 0)] = color_set[dot_class]
        io.imsave(os.path.join(out_gt_dir, img_name + '_img_with_dots.jpg'), img2)

        # Generate the Gaussian maps.
        # It is important to not do this for each class separately because this may result in intersections in the detection map
        mat_s_points = np.where(patch_label_arr_dots > 0)
        points = np.zeros((len(mat_s_points[0]), 2))
        print(points.shape)
        points[:, 0] = mat_s_points[1]
        points[:, 1] = mat_s_points[0]
        patch_label_arr_dots_custom_all, patch_label_arr_dots_custom = gaussian_filter_density(img2, points,
                                                                                                patch_label_arr_dots,
                                                                                                out_gt_dmap_filepath)

