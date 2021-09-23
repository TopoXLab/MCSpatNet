import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm as tqdm
import skimage.io as io
from skimage.measure import label
from sklearn.cluster import KMeans


device=torch.device('cuda')

def collect_features(model, simple_train_loader, feature_indx_list):

    features_list = [[0]*len(simple_train_loader)] # list will hold the features extracted from each input image for each cell
    coord_list = [[0]*len(simple_train_loader)] # list will hold coordinates of cells in  each input image
    img_name_list = [0]*len(simple_train_loader) # list will hold filename of each input image
    model.eval()
    with torch.no_grad():
        for i,(img,gt_dmap,gt_dots,img_name, padding) in enumerate(tqdm(simple_train_loader, disable=True)):
            # padding: padding added to the image to make sure it is a multiple of 16 (corresponding to 4 max pool layers)
            pad_y1  = padding[0].numpy()[0]
            pad_y2  = padding[1].numpy()[0]
            pad_x1  = padding[2].numpy()[0]
            pad_x2  = padding[3].numpy()[0]

            # set the image filename
            img_name_list[i]=img_name[0]
            img=img.to(device)

            # get the ground truth dot map for all cells without the padding
            gt_dots = gt_dots[:,:,pad_y1:gt_dots.shape[-2]-pad_y2,pad_x1:gt_dots.shape[-1]-pad_x2]
            gt_dots_all =  gt_dots.max(1)[0]
            gt_dots_all = gt_dots_all.detach().cpu().numpy().squeeze()

            # get the image features from the model
            et_dmap_lst, img_feat=model(img, feature_indx_list)
            img_feat = img_feat[:,:,2:-2,2:-2]
            img_feat = img_feat[:,:,pad_y1:img_feat.shape[-2]-pad_y2,pad_x1:img_feat.shape[-1]-pad_x2]
            img_feat = img_feat.squeeze().transpose((1,2,0))

            # set the cell coordinates
            points = np.where(gt_dots_all > 0)
            coord_list[0][i] = points
            if(len(points[0])==0):
                features_list[0][i] = None
                continue

            # set the cell features
            img_feat = img_feat[points]
            features_list[0][i] = img_feat

            del et_dmap_lst

    return features_list, coord_list, img_name_list
        
def collect_features_by_class(model, simple_train_loader, feature_indx_list, n_classes):

    features_list = [[0]*len(simple_train_loader) for i in range(n_classes)]    # imp to avoid referencing the same list in all entries
    coord_list = [[0]*len(simple_train_loader) for i in range(n_classes)] # imp to avoid referencing the same list in all entries
    img_name_list = ['']*len(simple_train_loader)
    model.eval()
    with torch.no_grad():
        for i,(img,gt_dmap,gt_dots,img_name, padding) in enumerate(tqdm(simple_train_loader, disable=True)):
            # padding: padding added to the image to make sure it is a multiple of 16 (corresponding to 4 max pool layers)
            pad_y1  = padding[0].numpy()[0]
            pad_y2  = padding[1].numpy()[0]
            pad_x1  = padding[2].numpy()[0]
            pad_x2  = padding[3].numpy()[0]

            # set the image filename
            img_name_list[i]=img_name[0]
            img=img.to(device)

            # get the ground truth dot map for all cells without the padding
            gt_dots = gt_dots[:,:,pad_y1:gt_dots.shape[-2]-pad_y2,pad_x1:gt_dots.shape[-1]-pad_x2]
            gt_dots = gt_dots.detach().cpu().numpy().squeeze()

            # get the image features from the model
            et_dmap_lst, img_feat=model(img, feature_indx_list)
            img_feat = img_feat[:,:,2:-2,2:-2]
            img_feat = img_feat[:,:,pad_y1:img_feat.shape[-2]-pad_y2,pad_x1:img_feat.shape[-1]-pad_x2]
            img_feat = img_feat.squeeze().transpose((1,2,0))

            # for each cell class, get the cell coordinates and center features
            for s in range(gt_dots.shape[0]):
                points = np.where(gt_dots[s] > 0)
                coord_list[s][i] = points
                if(len(points[0])==0):
                    features_list[s][i] = None
                    continue
                img_feat_s = img_feat[points]
                features_list[s][i] = img_feat_s

            del et_dmap_lst            
 
    return features_list, coord_list, img_name_list
    
def cluster(features_list, coord_list, n_clusters, prev_centroids):
    # For each class, get all features and do kmeans clustering, then use the fitted kmeans to get the pseudo clustering label for each cell
    cluster_centers_all = None
    pseudo_labels_list = [[0]*len(features_list[0]) for i in range(len(features_list))]
    for s in range(len(features_list)):
        features = None
        # Concatenate all features from cells in the current class
        for i in range(len(features_list[s])):
            if(features_list[s][i] is None):
                continue
            if(features is None):
                features = features_list[s][i]
            else:
                features = np.concatenate((features, features_list[s][i]), axis=0)

        # To have a more stable clustering, we initialize kmeans centroids with previous clustering centroids
        if(prev_centroids is None):
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features)
        else:
            kmeans = KMeans(n_clusters=n_clusters, init=prev_centroids[s*n_clusters:s*n_clusters+n_clusters]).fit(features)

        # Predict the cluster label for each cell
        for i in range(len(features_list[s])):
            if(features_list[s][i] is None):
                pseudo_labels_list[s][i] = None
                continue
            pseudo_labels_list[s][i] = kmeans.predict(features_list[s][i])
        if(cluster_centers_all is None):
            cluster_centers_all = kmeans.cluster_centers_
        else:
            cluster_centers_all = np.concatenate((cluster_centers_all, kmeans.cluster_centers_), axis=0)

    # return the cluster labels and the centroids
    return pseudo_labels_list, cluster_centers_all

def create_pseudo_lbl_gt(simple_train_loader, pseudo_labels_list, coord_list, img_name_list, n_clusters, out_dir):
    n_subclasses = len(pseudo_labels_list) * n_clusters # number of sub classes is number of cell classes * number of clusters
    for i,(img,gt_dmap,gt_dots,img_name, padding) in enumerate(tqdm(simple_train_loader, disable=True)):
        ''' 
            img: input image
            gt_dmap: ground truth map for cell classes (lymphocytes, epithelial/tumor, stromal) with dilated dots. This can be a binary mask or a density map ( in which case it will be converted to a binary mask)
            gt_dots: ground truth binary dot map for cell classes (lymphocytes, epithelial/tumor, stromal). 
            img_name: img filename
            padding: padding added to the image to make sure it is a multiple of 16 (corresponding to 4 max pool layers)
        '''
        pad_y1  = padding[0].numpy()[0]
        pad_y2  = padding[1].numpy()[0]
        pad_x1  = padding[2].numpy()[0]
        pad_x2  = padding[3].numpy()[0]
        # get the ground truth maps without the padding
        gt_dmap = gt_dmap[:,:,pad_y1:gt_dmap.shape[-2]-pad_y2,pad_x1:gt_dmap.shape[-1]-pad_x2]
        gt_dots = gt_dots[:,:,pad_y1:gt_dots.shape[-2]-pad_y2,pad_x1:gt_dots.shape[-1]-pad_x2]
        # Convert ground truth maps to binary mask (in case they were density maps)
        gt_dmap = gt_dmap > 0

        # Initialize the ground truth maps for the clustering sub-classes
        gt_dmap_all =  gt_dmap.max(1)[0]
        gt_dots_all =  gt_dots.max(1)[0] 
        gt_dmap_all = gt_dmap_all.detach().cpu().numpy().squeeze()
        gt_dots_all = gt_dots_all.detach().cpu().numpy().squeeze()
        gt_dots_subclasses = np.zeros((gt_dots_all.shape[0], gt_dots_all.shape[1], n_subclasses+1))
        gt_dmap_subclasses = np.zeros((gt_dots_all.shape[0], gt_dots_all.shape[1], n_subclasses+1))

        label_comp = label(gt_dmap_all)
        cci = 0
        for s in range(len(pseudo_labels_list)):
            pseudo_labels = pseudo_labels_list[s][i]
            if(pseudo_labels is None):
                continue
            points = coord_list[s][i]
            for c in range(n_clusters):
                cci += 1
                # Set the dot map for the cell-sub-cluster
                gt_map_tmp = np.zeros((gt_dots_subclasses.shape[0],gt_dots_subclasses.shape[1]))
                gt_map_tmp [(points[0][(pseudo_labels == c)], points[1][(pseudo_labels == c)])]=1
                gt_dots_subclasses[:,:,cci] = gt_map_tmp

                # Set the dilated dot map (mask) for the cell-sub-cluster.
                gt_map_tmp = np.zeros((gt_dmap_subclasses.shape[0],gt_dmap_subclasses.shape[1]))
                # Assign to each connected component the same label as the ground truth dot in that cell
                comp_in_cluster = label_comp[(points[0][(pseudo_labels == c)], points[1][(pseudo_labels == c)])]
                for comp in comp_in_cluster:
                    gt_map_tmp[label_comp==comp] = 1
                gt_dmap_subclasses[:,:,cci] = gt_map_tmp
                # Save map as image. Useful for debugging.
                io.imsave(os.path.join(out_dir, img_name_list[i].replace('.png','_gt_dmap_s'+str(s)+'_c'+str(c)+'.png')), (gt_map_tmp*255).astype(np.uint8))

        # Save generated ground truth maps for the current image
        gt_dots_subclasses.astype(np.uint8).dump(os.path.join(out_dir, img_name_list[i].replace('.png','_gt_dots.npy')))
        gt_dmap_subclasses.astype(np.uint8).dump(os.path.join(out_dir, img_name_list[i].replace('.png','.npy')))
        

def perform_clustering(model, simple_train_loader, n_clusters, n_classes, feature_indx_list, out_dir, prev_centroids):
    '''
        model: MCSpatNet model being trained
        simple_train_loader: data loader for training data to iterate over input
        n_clusters: number of clusters per class
        n_classes: number of cell classes
        feature_indx_list: features to use in clustering [ feature_code = {'decoder':0, 'cell-detect':1, 'class':2, 'subclass':3, 'k-cell':4} ]
        out_dir: directory path to output generated pseudo ground truth
        prev_centroids: previous epoch clustering centroids
    '''

    # Get the features to use for clustering
    if(n_classes > 1):
        features_list, coord_list, img_name_list = collect_features_by_class(model, simple_train_loader, feature_indx_list, n_classes)
    else:
        features_list, coord_list, img_name_list = collect_features(model, simple_train_loader, feature_indx_list)

    # Do the clustering: get the centroids for the new clusters and the pseudo ground truth labels
    pseudo_labels_list, centroids = cluster(features_list, coord_list, n_clusters, prev_centroids)

    # Save the pseudo ground truth labels to the file system to be able to use in training
    create_pseudo_lbl_gt(simple_train_loader, pseudo_labels_list, coord_list, img_name_list, n_clusters, out_dir)
    return centroids

