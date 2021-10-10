import os
import numpy as np
import glob
import sys
import scipy.spatial
from skimage import io;
from scipy.ndimage.filters import convolve   
from scipy import ndimage

# Configuration parameters
# data_dir contains ground truth and prediction, assumes test_vis_mcspat.py was run first
data_dir = '../MCSpatNet_eval/mcspatnet_consep_1_e100'

max_dist_thresh = 6 # will compute fscore at distance thresholds in range (1,max_dist_thresh) # mpp = 0.254 at 40x,  ppm at 20x = 1/(0.254*2),  mpp at 20x = 0.254*2 = 0.508, 6 px = 0.508*6 = 3.048 microns, , 30 px = 0.508*30=15.24 microns
color_set = {'tp':(0,162,232),'fp':(0,255,0),'fn':(255,255,0)} 

def calc(g_dot, e_dot, class_indx, img_indx, img_name):
    '''
        Calculates number of TP, FP, FN for class_indx at different distance thresholds.
        For a threshold t, a TP prediction is within t pixels from a ground truth prediction that was not previously processed.
    '''
    leafsize = 2048
    k = 50
    e_coords = np.where(e_dot > 0)
    # Build kdtree from prediction cell centers
    z = np.zeros((len(e_coords[0]),2))
    z[:,0] = e_coords[0]
    z[:,1] = e_coords[1]
    if(len(e_coords[0]) > 0):
        tree = scipy.spatial.KDTree(z, leafsize=leafsize)
        print('tree.data.shape', tree.data.shape)

    for dist_thresh in range(1,max_dist_thresh+1):
        img_f = np.zeros((e_dot.shape[0],e_dot.shape[1],3))
        print('class_indx', class_indx, 'thresh', dist_thresh, 'len(e_coords[0])',len(e_coords[0]))
        if(len(e_coords[0]) == 0): # case: no predictions
            for dist_thresh in range(1,max_dist_thresh+1):
                tp_img = 0
                fn_img = (g_dot > 0).sum()
                fp_img = 0
                fn[class_indx, dist_thresh] += fn_img
        else:
            tp_img = 0
            fn_img = 0
            fp_img = 0

            e_dot_processing = np.copy(e_dot)

            gt_points = np.where(g_dot > 0)
            ''' 
                Loop over ground truth points and find nearest prediction within threshold distance
                    If there is a match and the matching point exists in e_dot_processing, 
                        then this is a TP, remove from the matching point from e_dot_processing so that each prediction is matched only once.
                    Otherwise
                        This is a FN
                Remaining predictions in e_dot_processing are counted as FPs                
            '''
            for pi in range(len(gt_points[0])):
                p = [[gt_points[0][pi], gt_points[1][pi]]]
                distances, locations = tree.query(p, k=k,distance_upper_bound =dist_thresh)
                match = False
                for nn in range(min(k,len(locations[0]))):
                    if((len(locations[0]) > 0) and (locations[0][nn] < tree.data.shape[0]) and (e_dot_processing[int(tree.data[locations[0][nn]][0]),int(tree.data[locations[0][nn]][1])] > 0)):
                    #if((len(locations[0]) > 0) and (locations[0][nn] < tree.data.shape[0]) ):
                        tp[class_indx, dist_thresh] += 1
                        tp_img +=1
                        e_dot_processing[int(tree.data[locations[0][nn]][0]),int(tree.data[locations[0][nn]][1])] = 0
                        match = True
                        py = int(tree.data[locations[0][nn]][0])
                        px = int(tree.data[locations[0][nn]][1])
                        img_f[max(0,py-2):min(img_f.shape[0],py+3), max(0,px-2):min(img_f.shape[1],px+3)] = color_set['tp']
                        break
                if(not match):
                    fn[class_indx, dist_thresh] += 1
                    fn_img +=1
                    py = gt_points[0][pi]
                    px = gt_points[1][pi]
                    img_f[max(0,py-2):min(img_f.shape[0],py+3), max(0,px-2):min(img_f.shape[1],px+3)] = color_set['fn']

            fp[class_indx, dist_thresh] += e_dot_processing.sum()
            fp_img +=e_dot_processing.sum()
            fp_points = np.where(e_dot_processing > 0)
            for pi in range(len(fp_points[0])):
                py = fp_points[0][pi]
                px = fp_points[1][pi]
                img_f[max(0,py-2):min(img_f.shape[0],py+3), max(0,px-2):min(img_f.shape[1],px+3)] = color_set['fp']
            #io.imsave(os.path.join(out_dir, img_name+'_s'+str(class_indx)+'_f'+'_th'+str(dist_thresh)+'.png'), img_f.astype(np.uint8))
            print(img_name, 's',class_indx, 'thresh', dist_thresh, 'tp', tp_img, 'fp', fp_img, 'fn', fn_img)
            sys.stdout.flush();

        # Calculate the precision, recall, and fscore for current image and threshold
        if(tp_img + fp_img == 0):
            precision_img[class_indx, dist_thresh, img_indx] = 1
        else:
            precision_img[class_indx, dist_thresh, img_indx] = tp_img/(tp_img + fp_img)
        if(tp_img + fn_img == 0):
            recall_img[class_indx, dist_thresh, img_indx] = 1
        else:
            recall_img[class_indx, dist_thresh, img_indx] = tp_img/(tp_img + fn_img) # True pos rate
        if(precision_img[class_indx, dist_thresh, img_indx] + recall_img[class_indx, dist_thresh, img_indx] == 0):
            f1_img[class_indx, dist_thresh, img_indx] = 0
        else:
            f1_img[class_indx, dist_thresh, img_indx] = 2*(( precision_img[class_indx, dist_thresh, img_indx]*recall_img[class_indx, dist_thresh, img_indx] )/( precision_img[class_indx, dist_thresh, img_indx]+recall_img[class_indx, dist_thresh, img_indx] ))


def eval(data_dir, out_dir):
    '''
        Assumes ground truth dot maps for cell classes is in same directory as prediction files
        Ground truth dot maps has the naming <img name>_gt_dots_class.npy
        Prediction dot maps has the naming <img name>_centers_s<class indx>.npy for classification and <img name>_centers_allcells.npy for detection
    '''

    img_indx=-1
    with open(os.path.join(out_dir, 'out_distance_scores.txt'), 'a+') as log_file:
        for gt_filepath in gt_files:
            img_indx += 1
            print('gt_filepath',gt_filepath)
            sys.stdout.flush()
            img_name = os.path.basename(gt_filepath)[:-len('_gt_dots_class.npy')]
            g_dot_arr=np.load(gt_filepath, allow_pickle=True)

            # process cell classification
            for s in range(n_classes):
                e_soft_filepath = glob.glob(os.path.join(data_dir, img_name + '_*'+'centers_s'+str(s)+'.npy'))[0]
                print('e_soft_filepath', e_soft_filepath)
                class_indx = s
                g_dot = g_dot_arr[class_indx]
                #print('e_soft_filepath',e_soft_filepath)
                sys.stdout.flush()
                e_dot = np.load(e_soft_filepath, allow_pickle=True)
                e_dot_vis = ndimage.convolve(e_dot, np.ones((5,5)), mode='constant', cval=0.0)            
                #io.imsave(os.path.join(data_dir,img_name + '_centers_s0_et.png'),(e_dot_vis*255).astype(np.uint8))
                calc(g_dot, e_dot, class_indx, img_indx, img_name)


            # process cell detection
            e_soft_filepath = glob.glob(os.path.join(data_dir, img_name + '_*'+'centers_all*.npy'))[0]
            class_indx += 1
            g_dot = g_dot_arr.max(axis=0)
            #print('e_soft_filepath',e_soft_filepath)
            sys.stdout.flush()
            e_dot = np.load(e_soft_filepath, allow_pickle=True)
            calc(g_dot, e_dot, class_indx, img_indx, img_name)



        # tp.astype(np.int).dump(os.path.join(out_dir, 'tp.npy'))
        # fp.astype(np.int).dump(os.path.join(out_dir, 'fp.npy'))
        # fn.astype(np.int).dump(os.path.join(out_dir, 'fn.npy'))

        # Compute the precision, recall, and f-score for each class (class indx in range (0, n_classes-1)) and for detection task (class indx = n_classes) at each distance threshold in the range (1, max_dist_thresh)
        for class_indx in range(n_classes_out):
            for dist_thresh in range(1,max_dist_thresh+1):
                if(tp[class_indx, dist_thresh] + fp[class_indx, dist_thresh] == 0):
                    precision[class_indx, dist_thresh] = 1
                else:
                    precision[class_indx, dist_thresh] = tp[class_indx, dist_thresh]/(tp[class_indx, dist_thresh] + fp[class_indx, dist_thresh])
                if(tp[class_indx, dist_thresh] + fn[class_indx, dist_thresh] == 0):
                    recall[class_indx, dist_thresh] = 1
                else:
                    recall[class_indx, dist_thresh] = tp[class_indx, dist_thresh]/(tp[class_indx, dist_thresh] + fn[class_indx, dist_thresh]) # True pos rate
                if(precision[class_indx, dist_thresh] + recall[class_indx, dist_thresh] == 0):
                    f1[class_indx, dist_thresh] = 0
                else:
                    f1[class_indx, dist_thresh] = 2*((precision[class_indx, dist_thresh]*recall[class_indx, dist_thresh])/(precision[class_indx, dist_thresh]+recall[class_indx, dist_thresh]))

                print('class', class_indx, 'thresh', dist_thresh, 'prec', precision[class_indx, dist_thresh], 'recall', recall[class_indx, dist_thresh], 'fscore',f1[class_indx, dist_thresh])
                log_file.write("class {} thresh {} prec {} recall {} fscore {}\n".format(class_indx, dist_thresh, precision[class_indx, dist_thresh], recall[class_indx, dist_thresh], f1[class_indx, dist_thresh]))


            log_file.flush()

if __name__ == "__main__":
    out_dir= data_dir # can change the output directory
    n_classes=3 # number of cell classes
    n_classes_out = n_classes + 1 # output includes cell classes and cells detection

    # Initialize statistics variables
    tp = np.zeros((n_classes_out, max_dist_thresh + 1))
    fp = np.zeros((n_classes_out, max_dist_thresh + 1))
    fn = np.zeros((n_classes_out, max_dist_thresh + 1))
    precision = np.zeros((n_classes_out, max_dist_thresh + 1))
    recall = np.zeros((n_classes_out, max_dist_thresh + 1))
    f1 = np.zeros((n_classes_out, max_dist_thresh + 1))

    gt_files = glob.glob(os.path.join(data_dir, '*_gt_dots_class'+'.npy'))
    #gt_files = glob.glob(os.path.join(data_dir, '*test_1_gt_dots_class'+'.npy'))
    print('len(gt_files)',len(gt_files))

    precision_img = np.zeros((n_classes_out, max_dist_thresh + 1, len(gt_files)))
    recall_img = np.zeros((n_classes_out, max_dist_thresh + 1, len(gt_files)))
    f1_img = np.zeros((n_classes_out, max_dist_thresh + 1, len(gt_files)))

    eval(data_dir, out_dir)



