import numpy as np
import time
import torch
import torch.nn as nn
import os
from tqdm import tqdm as tqdm
import sys;
import math
import skimage.io as io
import cv2
from skimage import filters
from skimage.measure import label, moments
import glob

from model_arch import UnetVggMultihead
from my_dataloader_w_kfunc import CellsDataset
from my_dataloader import CellsDataset as CellsDataset_simple
from cluster_helper import *

checkpoints_root_dir = '../MCSpatNet_checkpoints' # The root directory for all training output.
checkpoints_folder_name = 'mcspatnet_consep_1' # The name of the folder that will be created under <checkpoints_root_dir> to hold output from current training instance.
model_param_path        = None;  # path of a previous checkpoint to continue training
clustering_pseudo_gt_root = '../MCSpatNet_epoch_subclasses'
train_data_root = '../MCSpatNet_datasets/CoNSeP_train'
test_data_root = '../MCSpatNet_datasets/CoNSeP_train'
train_split_filepath = './data_splits/consep/train_split.txt'
test_split_filepath = './data_splits/consep/val_split.txt'
epochs  = 300 # number of training epochs. Use 300 for CoNSeP dataset.


cell_code = {1:'lymphocyte', 2:'tumor', 3:'stromal'}

feature_code = {'decoder':0, 'cell-detect':1, 'class':2, 'subclass':3, 'k-cell':4}




if __name__=="__main__":

    # checkpoints_save_path: path to save checkpoints
    checkpoints_save_path   = os.path.join(checkpoints_root_dir, checkpoints_folder_name)
    cluster_tmp_out         = os.path.join(clustering_pseudo_gt_root, checkpoints_folder_name)

    if not os.path.exists(checkpoints_root_dir):
        os.mkdir(checkpoints_root_dir)

    if not os.path.exists(checkpoints_save_path):
        os.mkdir(checkpoints_save_path)

    if not os.path.exists(clustering_pseudo_gt_root):
        os.mkdir(clustering_pseudo_gt_root)

    if not os.path.exists(cluster_tmp_out):
        os.mkdir(cluster_tmp_out)


    # log_file_path: path to save log file
    i=1
    while(True):
        log_file_path = os.path.join(checkpoints_root_dir, checkpoints_folder_name, f'train_log_{i}.txt') 
        if(not os.path.exists(log_file_path)):
            break
        i +=1

    start_epoch             = 0  # To use if continuing training from a previous epoch loaded from model_param_path
    epoch_start_eval_prec   = 1 # After epoch_start_eval_prec epochs start to evaluate F-score of predictions on the validation set.
    restart_epochs_freq     = 50 # reset frequency for optimizer
    next_restart_epoch      = restart_epochs_freq + start_epoch
    gpu_or_cpu              ='cuda' # use cuda or cpu
    device=torch.device(gpu_or_cpu)
    seed                    = time.time()
    # print_frequency         = 1  # print frequency per epoch

    # Initialize log file
    log_file = open(log_file_path, 'a+')

    
    # Configure training dataset
    train_image_root = os.path.join(train_data_root, 'images')
    train_dmap_root = os.path.join(train_data_root, 'gt_custom') 
    train_dots_root = os.path.join(train_data_root, 'gt_custom') 
    train_dmap_subclasses_root = cluster_tmp_out
    train_dots_subclasses_root = train_dmap_subclasses_root
    train_kmap_root = os.path.join(train_data_root, 'k_func_maps') 

    # Configure validation dataset
    test_image_root = os.path.join(test_data_root, 'images')
    test_dmap_root = os.path.join(test_data_root, 'gt_custom')
    test_dots_root = os.path.join(test_data_root, 'gt_custom')
    test_dmap_subclasses_root = cluster_tmp_out
    test_dots_subclasses_root = test_dmap_subclasses_root
    test_kmap_root = os.path.join(test_data_root, 'k_func_maps') 
    

    dropout_prob = 0.2
    initial_pad = 126 # We add padding so that final output has same size as input since we do not use same padding conv.
    interpolate = 'False'
    conv_init = 'he'

    n_channels = 3
    n_classes = 3 # number of cell classes (lymphocytes, tumor, stromal)
    n_classes_out = n_classes + 1 # number of output classes = number of cell classes (lymphocytes, tumor, stromal) + 1 (for cell detection channel)
    class_indx = '1,2,3' # the index of the classes channels in the ground truth
    n_clusters = 5 # number of clusters per class
    n_classes2 = n_clusters * (n_classes) # number of output classes for the cell cluster classification

    lr  = 0.00005 # learning rate
    batch_size = 1
    prints_per_epoch=1 # print frequency per epoch

    # Initialize the range of the radii for the K function for each class
    r_step = 15
    r_range = range(0, 100, r_step)
    r_arr = np.array([*r_range])
    r_classes = len(r_range) # number of output channels for the K function for a single class
    r_classes_all = r_classes * (n_classes ) # number of output channels for the K function over all classes

    k_norm_factor = 100 # the maximum K-value (i.e. number of nearby cells at radius r) to normalize the K-func to [0,1]
    lamda_dice = 1;  # weight for dice loss for main output channels (cell detection + cell classification)
    lamda_subclasses = 1 # weight for dice loss for secondary output channels (cell cluster classification)
    lamda_k = 1 # weight for L1 loss for K function regression


    torch.cuda.manual_seed(seed)
    model=UnetVggMultihead(kwargs={'dropout_prob':dropout_prob, 'initial_pad':initial_pad, 'interpolate':interpolate, 'conv_init':conv_init, 'n_classes':n_classes, 'n_channels':n_channels, 'n_heads':4, 'head_classes':[1,n_classes,n_classes2, r_classes_all]})
    if(not (model_param_path is None)):
        model.load_state_dict(torch.load(model_param_path), strict=False);
        log_file.write('model loaded \n')        
        log_file.flush()
    model.to(device)

    # Initialize sigmoid layer for cell detection
    criterion_sig = nn.Sigmoid()
    # Initialize softmax layer for cell classification
    criterion_softmax = nn.Softmax(dim=1)
    # Initialize L1 loss for K function regression
    criterion_l1_sum = nn.L1Loss(reduction='sum')

    # Initialize Optimizer
    optimizer=torch.optim.Adam(list(model.final_layers_lst.parameters())+list(model.decoder.parameters())+list(model.bottleneck.parameters())+list(model.encoder.parameters()),lr)

    # Initialize training dataset loader
    train_dataset=CellsDataset(train_image_root,train_dmap_root,train_dots_root,class_indx,train_dmap_subclasses_root, train_dots_subclasses_root, train_kmap_root, split_filepath=train_split_filepath, phase='train', fixed_size=448, max_scale=16)
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)

    # Initialize validation dataset loader
    test_dataset=CellsDataset(test_image_root,test_dmap_root,test_dots_root,class_indx,test_dmap_subclasses_root, test_dots_subclasses_root, test_kmap_root, split_filepath=test_split_filepath,phase='test', fixed_size=-1, max_scale=16)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)

    # Initialize training dataset loader for clustering phase
    simple_train_dataset=CellsDataset_simple(train_image_root,train_dmap_root,train_dots_root,class_indx, phase='test', fixed_size=-1, max_scale=16, return_padding=True)
    simple_train_loader=torch.utils.data.DataLoader(simple_train_dataset,batch_size=batch_size,shuffle=False)


    # Use prints_per_epoch to get iteration number to generate sample output
    # print_frequency = len(train_loader)//prints_per_epoch;
    print_frequency_test = len(test_loader) // prints_per_epoch;

    best_epoch_filepath=None
    best_epoch=None
    best_f1_mean = 0
    best_prec_recall_diff = math.inf

    centroids = None
    for epoch in range(start_epoch,epochs):
        # If epoch already exists then skip
        epoch_files = glob.glob(os.path.join(checkpoints_save_path, 'mcspat_epoch_'+str(epoch)+"_*.pth"))
        if len(epoch_files) > 0:
            continue;
        # Cluster features at the beginning of each epoch
        print('epoch', epoch, 'start clustering')
        centroids = perform_clustering(model, simple_train_loader, n_clusters, n_classes, [feature_code['k-cell'], feature_code['subclass']], train_dmap_subclasses_root, centroids)
        print('epoch', epoch, 'end clustering')
                
        # Training phase
        model.train()
        log_file.write('epoch= ' + str(epoch) + '\n')
        log_file.flush()

        # Initialize variables for accumulating loss over the epoch
        epoch_loss=0
        train_count = 0
        # train_loss_k = 0
        # train_loss_dice = 0
        # train_count_k = 0


        for i,(img,gt_dmap,gt_dots,gt_dmap_subclasses,gt_dots_subclasses, gt_kmap,img_name) in enumerate(tqdm(train_loader)):
            ''' 
                img: input image
                gt_dmap: ground truth map for cell classes (lymphocytes, epithelial/tumor, stromal) with dilated dots. This can be a binary mask or a density map ( in which case it will be converted to a binary mask)
                gt_dots: ground truth binary dot map for cell classes (lymphocytes, epithelial/tumor, stromal). 
                gt_dmap_subclasses: ground truth map for cell clustering sub-classes with dilated dots. This can be a binary mask or a density map ( in which case it will be converted to a binary mask) 
                gt_dots_subclasses: ground truth binary dot map for cell clustering sub-classes. 
                gt_kmap: ground truth k-function map. At each cell center contains the cross k-functions centered at that cell. 
                img_name: img filename
            '''
            gt_kmap /= k_norm_factor # Normalize K functions ground truth
            img_name=img_name[0]
            train_count += 1

            img=img.to(device)
            # Convert ground truth maps to binary mask (in case they were density maps)
            gt_dmap = gt_dmap > 0
            gt_dmap_subclasses = gt_dmap_subclasses > 0
            # Get the detection ground truth maps from the classes ground truth maps
            gt_dmap_all =  gt_dmap.max(1)[0]
            gt_dots_all =  gt_dots.max(1)[0] 
            # Set datatype and move to GPU
            gt_dmap = gt_dmap.type(torch.FloatTensor)
            gt_dmap_all = gt_dmap_all.type(torch.FloatTensor)
            gt_dmap_subclasses = gt_dmap_subclasses.type(torch.FloatTensor)
            gt_kmap = gt_kmap.type(torch.FloatTensor)
            gt_dmap=gt_dmap.to(device)
            gt_dmap_all=gt_dmap_all.to(device)
            gt_dmap_subclasses=gt_dmap_subclasses.to(device)
            gt_kmap=gt_kmap.to(device)

            # forward propagation        
            et_dmap_lst=model(img)
            et_dmap_all=et_dmap_lst[0][:,:,2:-2,2:-2] # The cell detection prediction
            et_dmap_class=et_dmap_lst[1][:,:,2:-2,2:-2] # The cell classification prediction
            et_dmap_subclasses= et_dmap_lst[2][:,:,2:-2,2:-2] # The cell clustering sub-class prediction
            et_kmap=et_dmap_lst[3][:,:,2:-2,2:-2]**2   # The cross K-functions estimation

            # Apply K function loss only on the detection mask regions
            k_loss_mask = gt_dmap_all.clone()
            loss_l1_k = criterion_l1_sum(et_kmap*(k_loss_mask), gt_kmap*(k_loss_mask)) / (k_loss_mask.sum()*r_classes_all)

            # Apply Sigmoid and Softmax activations to the detection and classification predictions, respectively.
            et_all_sig = criterion_sig(et_dmap_all)
            et_class_sig = criterion_softmax(et_dmap_class)
            et_subclasses_sig = criterion_softmax(et_dmap_subclasses)

            # Compute Dice loss on the detection and classification predictions
            intersection = (et_class_sig * gt_dmap ).sum()
            union = (et_class_sig**2).sum() + (gt_dmap**2).sum()
            loss_dice_class =  1 - ((2 * intersection + 1) / (union + 1))

            intersection = (et_all_sig * gt_dmap_all.unsqueeze(0) ).sum()
            union = (et_all_sig**2).sum() + (gt_dmap_all.unsqueeze(0)**2).sum()
            loss_dice_all =  1 - ((2 * intersection + 1) / (union + 1))

            intersection = (et_subclasses_sig * gt_dmap_subclasses ).sum()
            union = (et_subclasses_sig**2).sum() + (gt_dmap_subclasses**2).sum()
            loss_dice_subclass =  1 - ((2 * intersection + 1) / (union + 1))

            loss_dice = loss_dice_class + loss_dice_all + lamda_subclasses * loss_dice_subclass
            # train_loss_dice += loss_dice.item()

            # Add up the dice loss and the K function L1 loss. The K function can be NAN especially in the beginning of training. Do not add to loss if it is NAN.
            loss = (lamda_dice * loss_dice )
            if(not math.isnan(loss_l1_k.item())):
                loss += loss_l1_k * lamda_k
                # train_count_k += 1
                # train_loss_k += loss_l1_k.item()

            # Backpropagate loss
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            log_file.write("epoch: "+str(epoch)+ "  i: "+str(i)+"   loss_dice: "+str(loss_dice.item()) + "   loss_l1_k:" + str(loss_l1_k.item()) + '\n')
            log_file.flush()


        log_file.write("epoch: " + str(epoch) + " train loss: "+ str(epoch_loss/train_count)+ '\n')
        log_file.flush()
        epoch_loss = epoch_loss/train_count

        #break

        # Testing phase on Validation Set
        model.eval()
        err=np.array([0 for s in range(n_classes_out)])
        loss_val = 0
        loss_val_k_wo_nan = 0
        loss_val_k = 0
        loss_val_dice = 0
        loss_val_dice2 = 0
        tp_count_all = np.zeros((n_classes_out))
        fp_count_all = np.zeros((n_classes_out))
        fn_count_all = np.zeros((n_classes_out))
        test_count_k = 0
        for i,(img,gt_dmap,gt_dots,gt_dmap_subclasses,gt_dots_subclasses, gt_kmap,img_name) in enumerate(tqdm(test_loader)):
            ''' 
                img: input image
                gt_dmap: ground truth map for cell classes (lymphocytes, epithelial/tumor, stromal) with dilated dots. This can be a binary mask or a density map ( in which case it will be converted to a binary mask)
                gt_dots: ground truth binary dot map for cell classes (lymphocytes, epithelial/tumor, stromal). 
                gt_dmap_subclasses: ground truth map for cell clustering sub-classes with dilated dots. This can be a binary mask or a density map ( in which case it will be converted to a binary mask) 
                gt_dots_subclasses: ground truth binary dot map for cell clustering sub-classes. 
                gt_kmap: ground truth k-function map. At each cell center contains the cross k-functions centered at that cell. 
                img_name: img filename
            '''
            gt_kmap /= k_norm_factor # Normalize K functions ground truth
            img_name=img_name[0]
            img=img.to(device)
            # Convert ground truth maps to binary masks (in case they were density maps)
            gt_dmap = gt_dmap > 0
            # Get the detection ground truth maps from the classes ground truth maps
            gt_dmap_all =  gt_dmap.max(1)[0]
            gt_dots_all =  gt_dots.max(1)[0]
            # Set datatype and move to GPU
            gt_dmap = gt_dmap.type(torch.FloatTensor)
            gt_dmap_all = gt_dmap_all.type(torch.FloatTensor)
            gt_kmap = gt_kmap.type(torch.FloatTensor)
            gt_kmap=gt_kmap.to(device)
            k_loss_mask = gt_dmap_all.clone().to(device)      # Apply K-function loss only on the dilated dots mask

            # Convert ground truth maps to numpy arrays
            gt_dots = gt_dots.detach().cpu().numpy()
            gt_dots_all = gt_dots_all.detach().cpu().numpy()
            gt_dmap = gt_dmap.detach().cpu().numpy()
            gt_dmap_all = gt_dmap_all.detach().cpu().numpy()

            # forward Propagation
            et_dmap_lst=model(img)
            et_dmap_all=et_dmap_lst[0][:,:,2:-2,2:-2] # The cell detection prediction
            et_dmap_class=et_dmap_lst[1][:,:,2:-2,2:-2] # The cell classification prediction
            et_dmap_subclasses= et_dmap_lst[2][:,:,2:-2,2:-2] # The cell clustering sub-class prediction
            et_kmap=et_dmap_lst[3][:,:,2:-2,2:-2]**2   # The cross K-functions estimation

            # Apply Sigmoid and Softmax activations to the detection and classification predictions, respectively.
            et_all_sig = criterion_sig(et_dmap_all).detach().cpu().numpy()
            et_class_sig = criterion_softmax(et_dmap_class).detach().cpu().numpy()

            # Apply K function loss only on the detection mask regions
            loss_l1_k = criterion_l1_sum(et_kmap*(k_loss_mask), gt_kmap*(k_loss_mask)) / (k_loss_mask.sum()*r_classes_all)

            # Save sample output predictions
            if(i % print_frequency_test == 0):
                io.imsave(os.path.join(checkpoints_save_path, 'test'+ '_indx'+str(i)+'_img'+'.png'), (img.squeeze().detach().cpu().numpy()*255).transpose(1,2,0).astype(np.uint8));
                for s in range(n_classes):
                    io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_test'+ '_indx'+str(i)+'_likelihood'+'_s'+str(s)+'.png'), (et_class_sig[:,s,:,:]*255).squeeze().astype(np.uint8));
                    io.imsave(os.path.join(checkpoints_save_path, 'test'+ '_indx'+str(i)+'_gt'+'_s'+str(s)+'.png'), (gt_dmap[:,s,:,:]*255).squeeze().astype(np.uint8));
                io.imsave(os.path.join(checkpoints_save_path, 'epoch'+str(epoch)+ '_test'+ '_indx'+str(i)+'_likelihood'+'_all'+'.png'), (et_all_sig*255).squeeze().astype(np.uint8));
                io.imsave(os.path.join(checkpoints_save_path, 'test'+ '_indx'+str(i)+'_gt'+'_all'+'.png'), (gt_dmap_all*255).squeeze().astype(np.uint8));

            # Accumulate K-function test losses
            loss_val_k += loss_l1_k.item()
            if(not math.isnan(loss_l1_k.item())):
                loss_val_k_wo_nan += loss_l1_k.item()
                test_count_k += 1

            # Compute Dice loss on the detection and classification predictions
            intersection = (et_class_sig * gt_dmap ).sum()
            union = (et_class_sig**2).sum() + (gt_dmap**2).sum()
            loss_dice_class =  1 - ((2 * intersection + 1) / (union + 1))

            intersection = (et_all_sig * gt_dmap_all ).sum()
            union = (et_all_sig**2).sum() + (gt_dmap_all**2).sum()
            loss_dice_all =  1 - ((2 * intersection + 1) / (union + 1))

            loss_dice = (loss_dice_class + loss_dice_all).item()
            loss_val_dice += loss_dice

            print('epoch', epoch, 'test', i, 'loss_l1_k', str(loss_l1_k.item()), 'loss_dice', str(loss_dice))

            # Calculate F-score if epoch >= epoch_start_eval_prec
            if(epoch >= epoch_start_eval_prec):
                # Apply a 0.5 threshold on detection output and convert to binary mask
                e_hard = filters.apply_hysteresis_threshold(et_all_sig.squeeze(), 0.5, 0.5)            
                e_hard2 = (e_hard > 0).astype(np.uint8)
                e_hard2_all = e_hard2.copy() 

                # Get predicted cell centers by finding center of contours in binary mask
                e_dot = np.zeros((img.shape[-2], img.shape[-1]))
                contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for idx in range(len(contours)):
                    contour_i = contours[idx]
                    M = cv2.moments(contour_i)
                    if(M['m00'] == 0):
                        continue;
                    cx = round(M['m10'] / M['m00'])
                    cy = round(M['m01'] / M['m00'])
                    e_dot[cy, cx] = 1
                e_dot_all = e_dot.copy()

                tp_count = 0 # initialize number of true positives
                fp_count = 0 # initialize number of false positives
                fn_count = 0 # initialize number of false negatives
                # Init g_dot_vis to contain all cell detections ground truth dots
                g_dot_vis = gt_dots_all.copy().squeeze()
                # Get connected components in the predicted detection binary map
                e_hard2_comp = label(e_hard2)
                e_hard2_comp_all = e_hard2_comp.copy()
                # For each connected component, if it interests with a grount truth dot then it is a TP, otherwise it is a FP.
                # If it is a TP, remove it from g_dot_vis.
                # Note: if more than one ground truth dot interests, then only one is a TP.
                for l in range(1, e_hard2_comp.max()+1):
                    e_hard2_comp_l = (e_hard2_comp == l)
                    M = moments(e_hard2_comp_l)
                    (y,x) = int(M[1, 0] / M[0, 0]), int(M[0, 1] / M[0, 0])
                    if ((e_hard2_comp_l * g_dot_vis).sum()>0): # true pos
                        tp_count += 1
                        (yg,xg) = np.where((e_hard2_comp_l * g_dot_vis) > 0)
                        yg = yg[0]
                        xg = xg[0]
                        g_dot_vis[yg,xg] = 0 
                    else: #((e_hard2_comp_l * g_dot_vis).sum()==0): # false pos
                        fp_count += 1
                # Remaining cells in g_dot_vis are False Negatives.
                fn_points = np.where(g_dot_vis > 0)
                fn_count = len(fn_points[0])

                # Update TP, FP, FN counts for detection with counts from current image predictions
                tp_count_all[-1] = tp_count_all[-1] + tp_count
                fp_count_all[-1] = fp_count_all[-1] + fp_count
                fn_count_all[-1] = fn_count_all[-1] + fn_count

                # Get predicted cell classes
                et_class_argmax = et_class_sig.squeeze().argmax(axis=0)
                e_hard2_all = e_hard2.copy()
                # For each class get the TP, FP, FN counts similar to previous detection code.
                for s in range(n_classes):
                    g_count = gt_dots[0,s,:,:].sum()

                    e_hard2 = (et_class_argmax == s)  
                
                    e_dot = e_hard2 * e_dot_all  

                    g_dot = gt_dots[0,s,:,:].squeeze()

                    tp_count = 0
                    fp_count = 0
                    fn_count = 0
                    g_dot_vis = g_dot.copy()
                    e_dots_tuple = np.where(e_dot > 0)
                    for idx in range(len(e_dots_tuple[0])):
                        cy=e_dots_tuple[0][idx]
                        cx=e_dots_tuple[1][idx]
                        l = e_hard2_comp_all[cy, cx]
                        e_hard2_comp_l = (e_hard2_comp == l)
                        if ((e_hard2_comp_l * g_dot_vis).sum()>0): # true pos
                            tp_count += 1
                            (yg,xg) = np.where((e_hard2_comp_l * g_dot_vis) > 0)
                            yg = yg[0]
                            xg = xg[0]
                            g_dot_vis[yg,xg] = 0 
                        else: #((e_hard2_comp_l * g_dot_vis).sum()==0): # false pos
                            fp_count += 1
                    fn_points = np.where(g_dot_vis > 0)
                    fn_count = len(fn_points[0])


                    tp_count_all[s] = tp_count_all[s] + tp_count
                    fp_count_all[s] = fp_count_all[s] + fp_count
                    fn_count_all[s] = fn_count_all[s] + fn_count


            del img,gt_dmap,gt_dmap_all,gt_dmap_subclasses, gt_kmap, et_dmap_all, et_dmap_class, et_kmap,gt_dots


        saved = False

        precision_all = np.zeros((n_classes_out))
        recall_all = np.zeros((n_classes_out))
        f1_all = np.zeros((n_classes_out))
        if(epoch >= epoch_start_eval_prec):
            count_all = tp_count_all.sum() + fn_count_all.sum()
            for s in range(n_classes_out):
                if(tp_count_all[s] + fp_count_all[s] == 0):
                    precision_all[s] = 1
                else:
                    precision_all[s] = tp_count_all[s]/(tp_count_all[s] + fp_count_all[s])
                if(tp_count_all[s] + fn_count_all[s] == 0):
                    recall_all[s] = 1
                else:
                    recall_all[s] = tp_count_all[s]/(tp_count_all[s] + fn_count_all[s])
                if(precision_all[s]+recall_all[s] == 0):
                    f1_all[s] = 0
                else:
                    f1_all[s] = 2*(precision_all[s] *recall_all[s])/(precision_all[s]+recall_all[s])
                print_msg = f'epoch {epoch} s {s} precision_all {precision_all[s]} recall_all {recall_all[s]} f1_all {f1_all[s]}'
                print(print_msg)
                log_file.write(print_msg+'\n')
                log_file.flush()
            print_msg = f'epoch {epoch} all precision_all {precision_all.mean()} recall_all {recall_all.mean()} f1_all {f1_all.mean()}'
            print(print_msg)
            log_file.write(print_msg+'\n')
            log_file.flush()
            print_msg = f'epoch {epoch} classes precision_all {precision_all[:-1].mean()} recall_all {recall_all[:-1].mean()} f1_all {f1_all[:-1].mean()}'
            print(print_msg)
            log_file.write(print_msg+'\n')
            log_file.flush()


        # Check if this is the best epoch so far based on fscore on validation set
        model_save_postfix = ''
        is_best_epoch = False
        # if (f1_all.mean() > best_f1_mean):
        if (f1_all.mean() - best_f1_mean >= 0.005):
            model_save_postfix += '_f1'
            best_f1_mean = f1_all.mean()
            best_prec_recall_diff = abs(recall_all.mean()-precision_all.mean())
            is_best_epoch = True
        elif ((abs(f1_all.mean() - best_f1_mean) < 0.005) # a slightly lower f score but smaller gap between precision and recall
                and abs(recall_all.mean()-precision_all.mean()) < best_prec_recall_diff):
            model_save_postfix += '_pr-diff'
            best_f1_mean = f1_all.mean()
            best_prec_recall_diff = abs(recall_all.mean()-precision_all.mean())
            is_best_epoch = True
        # if (recall_all.mean() > best_recall_mean):
        #     model_save_postfix += '_rec'
        #     best_recall_mean = recall_all.mean()
        #     is_best_epoch = True


        # Save checkpoint if it is best so far
        if((saved == False) and (model_save_postfix != '')):
            print('epoch', epoch, 'saving')
            new_epoch_filepath = os.path.join(checkpoints_save_path, 'mcspat_epoch_'+str(epoch)+model_save_postfix+".pth")
            torch.save(model.state_dict(), new_epoch_filepath ) # save only if get better error
            centroids.dump(os.path.join(checkpoints_save_path, 'epoch{}_centroids.npy'.format(epoch)))
            saved = True
            print_msg = f'epoch {epoch} saved.'
            print(print_msg)
            log_file.write(print_msg+'\n')
            log_file.flush()
            if(is_best_epoch):
                best_epoch_filepath = new_epoch_filepath
                best_epoch = epoch

        # Adam optimizer needs resetting to avoid parameters learning rates dying
        sys.stdout.flush();
        if((epoch >= next_restart_epoch) and not(best_epoch_filepath is None)):
            next_restart_epoch = epoch + restart_epochs_freq
            model.load_state_dict(torch.load(best_epoch_filepath), strict=False);
            model.to(device)
            optimizer=torch.optim.Adam(list(model.final_layers_lst.parameters())+list(model.decoder.parameters())+list(model.bottleneck.parameters())+list(model.encoder.parameters()),lr) 

    log_file.close()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    