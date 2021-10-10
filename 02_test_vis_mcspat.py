import os
import numpy as np
from skimage import io;
import cv2 ;
import sys;
from skimage.measure import label, moments
from skimage import filters
from tqdm import tqdm as tqdm
import torch
import torch.nn as nn
import glob

from model_arch import UnetVggMultihead
from my_dataloader import CellsDataset

checkpoints_root_dir = '../MCSpatNet_checkpoints' # The root directory for all training output.
checkpoints_folder_name = 'mcspatnet_consep_1' # The name of the current training output folder under <checkpoints_root_dir>.
eval_root_dir = '../MCSpatNet_eval'
epoch=100 # the epoch to test
visualize=True # whether to output a visualization of the prediction
test_data_root = '../MCSpatNet_datasets/CoNSeP_test'
test_split_filepath = None

if __name__=="__main__":

    # Initializations

    #0: Lymphocyte: blue
    #1: Tumor: red
    #2: Other: yellow
    color_set = {0:(0,162,232),1:(255,0,0),2:(0,255,0)} 

    # model checkpoint and output configuration parameters
    models_root_dir = os.path.join(checkpoints_root_dir, checkpoints_folder_name)
    out_dir = os.path.join(eval_root_dir, checkpoints_folder_name+f'_e{epoch}') 

    if(not os.path.exists(eval_root_dir)):
        os.mkdir(eval_root_dir)

    if(not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    # data configuration parameters
    test_image_root = os.path.join(test_data_root, 'images')
    test_dmap_root = os.path.join(test_data_root, 'gt_custom')
    test_dots_root = os.path.join(test_data_root, 'gt_custom')

    # Model configuration parameters
    gt_multiplier = 1    
    gpu_or_cpu='cuda' # use cuda or cpu
    dropout_prob = 0
    initial_pad = 126
    interpolate = 'False'
    conv_init = 'he'
    n_classes = 3
    n_classes_out = n_classes + 1
    class_indx = '1,2,3'
    class_weights = np.array([1,1,1]) 
    n_clusters = 5
    n_classes2 = n_clusters * (n_classes)

    r_step = 15
    r_range = range(0, 100, r_step)
    r_arr = np.array([*r_range])
    r_classes = len(r_range)
    r_classes_all = r_classes * (n_classes )

    thresh_low = 0.5
    thresh_high = 0.5
    size_thresh = 5


    device=torch.device(gpu_or_cpu)
    model=UnetVggMultihead(kwargs={'dropout_prob':dropout_prob, 'initial_pad':initial_pad, 'interpolate':interpolate, 'conv_init':conv_init, 'n_classes':n_classes, 'n_channels':3, 'n_heads':4, 'head_classes':[1,n_classes,n_classes2, r_classes_all]})
    model.to(device)
    criterion_sig = nn.Sigmoid() # initialize sigmoid layer
    criterion_softmax = nn.Softmax(dim=1) # initialize sigmoid layer
    test_dataset=CellsDataset(test_image_root,test_dmap_root,test_dots_root,class_indx, split_filepath=test_split_filepath,phase='test', fixed_size=-1, max_scale=16)
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)


    print('thresh', thresh_low, thresh_high)

    # Load model
    print('test epoch ' + str(epoch) )
    model_files = glob.glob(os.path.join(models_root_dir, 'mcspat_epoch_'+str(epoch)+'_*.pth'))
    model_files2 = glob.glob(os.path.join(models_root_dir, '*epoch_'+str(epoch)+'_*.pth'))
    if((model_files == None) or (len(model_files)==0)):
        if((model_files2 == None) or (len(model_files2)==0)):
            print('not found ', 'mcspat_epoch_'+str(epoch) )
            exit()
        else:
            model_param_path = model_files2[0]
    else:
        model_param_path = model_files[0]

    sys.stdout.flush();
    model.load_state_dict(torch.load(model_param_path), strict=True);
    model.to(device)
    model.eval()

    with torch.no_grad():

        for i,(img,gt_dmap,gt_dots,img_name) in enumerate(tqdm(test_loader, disable=True)):
            img_name = img_name[0]
            sys.stdout.flush();

            # Forward Propagation
            img=img.to(device)
            et_dmap_lst=model(img)
            et_dmap_all=et_dmap_lst[0][:,:,2:-2,2:-2]
            et_dmap_class=et_dmap_lst[1][:,:,2:-2,2:-2]
            et_dmap_subclasses= et_dmap_lst[2][:,:,2:-2,2:-2]
            et_kmap=et_dmap_lst[3][:,:,2:-2,2:-2]**2


            gt_dmap = gt_dmap > 0
            gt_dmap_all =  gt_dmap.max(1)[0].detach().cpu().numpy()
            gt_dots_all =  gt_dots.max(1)[0].detach().cpu().numpy().squeeze()
            gt_dots = gt_dots.detach().cpu().numpy()

            et_all_sig = criterion_sig(et_dmap_all).detach().cpu().numpy()
            et_class_sig = criterion_softmax(et_dmap_class).detach().cpu().numpy()

            img = img.detach().cpu().numpy().squeeze().transpose(1,2,0)*255
            img_centers_all = img.copy()
            img_centers_all_gt = img.copy()

            img_centers_all_all = img.copy()
            img_centers_all_all_gt = img.copy()


            # begin: eval detection all
            g_count = gt_dots_all.sum()

            # Get connected components in the prediction and apply a small size threshold
            e_hard = filters.apply_hysteresis_threshold(et_all_sig.squeeze(), thresh_low, thresh_high)
            e_hard2 = (e_hard > 0).astype(np.uint8)
            comp_mask = label(e_hard2)
            e_count = comp_mask.max()
            s_count=0
            if(size_thresh > 0):
                for c in range(1,comp_mask.max()+1):
                    s = (comp_mask == c).sum()
                    if(s < size_thresh):
                        e_count -=1
                        s_count +=1
                        e_hard2[comp_mask == c] = 0
            e_hard2_all = e_hard2.copy()

            # Get centers of connected components in the prediction
            e_dot = np.zeros((img.shape[0], img.shape[1]))
            e_dot_vis = np.zeros((img.shape[0], img.shape[1]))
            contours, hierarchy = cv2.findContours(e_hard2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for idx in range(len(contours)):
                contour_i = contours[idx]
                M = cv2.moments(contour_i)
                if(M['m00'] == 0):
                    continue;
                cx = round(M['m10'] / M['m00'])
                cy = round(M['m01'] / M['m00'])
                e_dot_vis[cy-1:cy+1, cx-1:cx+1] = 1
                e_dot[min(cy, e_dot.shape[0]-1), min(cx, e_dot.shape[1]-1)] = 1
                img_centers_all_all[cy-3:cy+3, cx-3:cx+3,:] = (0,0,0)
            e_dot_all = e_dot.copy()
            gt_centers = np.where(gt_dots_all > 0)
            for idx in range(len(gt_centers[0])):
                cx = gt_centers[1][idx]
                cy = gt_centers[0][idx]
                img_centers_all_all_gt[cy-3:cy+3, cx-3:cx+3,:] = (0,0,0)

            e_dot.astype(np.uint8).dump(
                os.path.join(out_dir, img_name.replace('.png',  '_centers' + '_all' + '.npy')))
            if(visualize):
                #io.imsave(os.path.join(out_dir, img_name.replace('.png','_centers'+'_allcells' +'.png')), (e_dot_vis*255).astype(np.uint8))
                io.imsave(os.path.join(out_dir, img_name.replace('.png','_centers'+'_det' +'_overlay.png')), (img_centers_all_all).astype(np.uint8))
                #io.imsave(os.path.join(out_dir, img_name.replace('.png','_allcells' +'_hard.png')), (e_hard2*255).astype(np.uint8))

            # end: eval detection all

            # begin: eval classification
            et_class_argmax = et_class_sig.squeeze().argmax(axis=0)
            e_hard2_all = e_hard2.copy()

            for s in range(n_classes):
                g_count = gt_dots[0,s,:,:].sum()

                e_hard2 = (et_class_argmax == s)

                # Filter the predicted detection dot map by the current class predictions
                e_dot = e_hard2 * e_dot_all
                e_count = e_dot.sum()

                g_dot = gt_dots[0,s,:,:].squeeze()
                e_dot_vis = np.zeros(g_dot.shape)
                e_dots_tuple = np.where(e_dot > 0)
                for idx in range(len(e_dots_tuple[0])):
                    cy=e_dots_tuple[0][idx]
                    cx=e_dots_tuple[1][idx]
                    img_centers_all[cy-3:cy+3, cx-3:cx+3,:] = color_set[s]


                gt_centers = np.where(g_dot > 0)
                for idx in range(len(gt_centers[0])):
                    cx = gt_centers[1][idx]
                    cy = gt_centers[0][idx]
                    img_centers_all_gt[cy-3:cy+3, cx-3:cx+3,:] = color_set[s]

                e_dot.astype(np.uint8).dump(os.path.join(out_dir, img_name.replace('.png', '_centers' + '_s' + str(s) + '.npy')))
                #if(visualize):
                #    io.imsave(os.path.join(out_dir, img_name.replace('.png','_likelihood_s'+ str(s)+'.png')), (et_class_sig.squeeze()[s]*255).astype(np.uint8));
            # end: eval classification


            et_class_sig.squeeze().astype(np.float16).dump(
                os.path.join(out_dir, img_name.replace('.png', '_likelihood_class' + '.npy')))
            et_all_sig.squeeze().astype(np.float16).dump(
                os.path.join(out_dir, img_name.replace('.png', '_likelihood_all' + '.npy')))
            gt_dots.squeeze().astype(np.uint8).dump(
                os.path.join(out_dir, img_name.replace('.png', '_gt_dots_class' + '.npy')))
            gt_dots_all.squeeze().astype(np.uint8).dump(
                os.path.join(out_dir, img_name.replace('.png', '_gt_dots_all' + '.npy')))
            if(visualize):
                io.imsave(os.path.join(out_dir, img_name.replace('.png','_centers'+'_class_overlay' +'.png')), (img_centers_all).astype(np.uint8))
                io.imsave(os.path.join(out_dir, img_name.replace('.png','_gt_centers'+'_class_overlay'+'.png')), (img_centers_all_gt).astype(np.uint8))
                io.imsave(os.path.join(out_dir, img_name), (img).astype(np.uint8))
                #io.imsave(os.path.join(out_dir, img_name.replace('.png','_likelihood_all'+'.png')), (et_all_sig.squeeze()*255).astype(np.uint8));

            del img,gt_dots




