
### Model Training

1. Go to the repository root folder:
	
	`cd ..`

2. Edit `01_train_mcspat.py` <br/>
Set the variables: 
	- `checkpoints_root_dir `: The root directory for all training output. <br/>
	- `checkpoints_folder_name`: The training output folder name. It will have the path `<checkpoints_root_dir>/<checkpoints_folder_name>`. <br/>
	- `model_param_path`: Path of a previous checkpoint to continue training from. <br/>
	- `clustering_pseudo_gt_root`: The root directory for all pseudo ground truth clustering labels that are generated during training. The current training clustering pseudo labels will be saved to `<clustering_pseudo_gt_root>/<checkpoints_folder_name>` <br/>
	- `train_data_root`: Root of the training dataset. <br/>
	- `test_data_root`: Root of the validation dataset. <br/>
	- `train_split_filepath`: path to the text file containing the files to include in training. If set to None will use all the images in the training images folder. <br/>
	- `test_split_filepath`: path to the text file containing the files to include in validation. If set to None will use all the images in the validation images folder. <br/>
	
 
	Default values are:

	    checkpoints_root_dir = '../MCSpatNet_checkpoints' 
		checkpoints_folder_name = 'mcspatnet_consep_1'
		model_param_path = None
		clustering_pseudo_gt_root = '../MCSpatNet_epoch_subclasses'
		train_data_root = '../MCSpatNet_datasets/CoNSeP_train'
		test_data_root = '../MCSpatNet_datasets/CoNSeP_train'
		train_split_filepath = './data_splits/consep/train_split.txt'
		test_split_filepath = './data_splits/consep/val_split.txt'

 
6. Start training
		
		CUDA_VISIBLE_DEVICES='1' nohup python 01_train_mcspat.py > tmp_log.txt &

	Change `CUDA_VISIBLE_DEVICES` and `tmp_log.txt` as appropriate.
 
### Model Testing

1. Go to the repository root folder.

2. Edit `02_test_vis_mcspat.py` <br/>
Set the variables: 
	- `checkpoints_root_dir `: The root directory for all training output. <br/>
	- `checkpoints_folder_name`: The training output folder name. The saved checkpoint is in the folder `<checkpoints_root_dir>/<checkpoints_folder_name>`. <br/>
	- `eval_root_dir`: The root directory for all test output. The predictions will be saved in `<eval_root_dir>/<checkpoints_folder_name>_e<epoch>`. <br/>
	- `epoch`: The checkpoint epoch to test.
	- `visualize`: Boolean indicating whether to output a visualization of the prediction. <br/>
	- `test_data_root`: Root of the test dataset. <br/>
	- `test_split_filepath`: path to the text file containing the files to include in test. If set to None will use all the images in the test images folder. <br/>
	
 
	Default values are:

	    checkpoints_root_dir = '../MCSpatNet_checkpoints' 
		checkpoints_folder_name = 'mcspatnet_consep_1'
		eval_root_dir = '../MCSpatNet_eval'
		epoch = 100
		visualize = True
		test_data_root = '../MCSpatNet_datasets/CoNSeP_test'
		test_split_filepath = None

3. Run `02_test_vis_mcspat.py`

		CUDA_VISIBLE_DEVICES='1' nohup python 02_test_vis_mcspat.py > tmp_test_log.txt &

	The output is the prediction with visualization (optional).

	- `<img_name>_gt_dots_class.npy`: The ground truth classification dot maps.  
	- `<img_name>_gt_dots_all.npy`: The ground truth detection dot map.  
	- `<img_name>_likelihood_class.npy`: The prediction classification likelihood maps. 
	- `<img_name>_likelihood_all.npy`: The prediction detection likelihood map. 
	- `<img_name>_centers_s<class id>.npy`: The prediction classification dot map for each cell type. (default: 0=inflammatory, 1=epithelial, 2=stromal) 
	- `<img_name>_centers_all.npy`: The prediction detection dot map.  
	
	(Optional) If visualization is set to True: 
	- `<img_name>.png`: input image
	- `<img_name>_centers_det_overlay.png`: visualization of the predicted cell detection overlaid on the image. 
	- `<img_name>_centers_class_overlay.png`: visualization of the predicted cell classification overlaid on the image. 
	- `<img_name>_gt_centers_class_overlay.png`: visualization of the ground truth cell classification overlaid on the image. 

4. Edit `03_eval_localization_fscore.py`<br/>
Set the variables: 
	- `data_dir `: The path of the prediction output from running `02_test_vis_mcspat.py`. <br/>
	- `max_dist_thresh`: evaluate with distance thresholds in `[1-<max_dist_thresh>]`. The distance threshold is the distance between a prediction center and a ground truth dot to regard as true positive. It is in pixels.

 
	Default value is:

	    data_dir = '../MCSpatNet_eval/mcspatnet_consep_1_e100'
		max_dist_thresh = 6
 
5. Run `03_eval_localization_fscore.py`

		python 03_eval_localization_fscore.py 

	Outputs the file `<data_dir>/out_distance_scores.txt` contains the precision, recall, and f-score for classification and detection for distance thresholds in `[1-<max_dist_thresh>]`.
