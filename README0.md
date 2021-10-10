# MCSpatNet
Repository for ICCV2021 MCSpatNet: Multi-Class Cell Detection Using Spatial Context Representation

### Set Up Environment

The code was tested with docker environment based on nvidia container for pytorch [release 18.09](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_18.09.html).  
with the following installations:  
- python 3.6.5  
- pytorch 0.5.0a0 (higher version such as 1.0.0 or 1.4.0 are expected to work as well)  
- Numpy 1.19.4  
- scikit-image 0.15.0  
- OpenCV 4.1.0 (we are only using basic functions, other versions may just as well)   
- Scipy 1.1.0  
- Pillow 6.1.0  
- tqdm 4.25.0  
and for the generation of K function maps for training:    
- R 4.0.3  (with the spatstat package installed)  
- Pandas 1.1.5
- pyper  
- rpy2 


The docker environment is available here:   
    
	docker pull shahira/pytorch_plus_r 


update brca 1_generate_dot_maps to <br/>
- rename solid to binary <br/>
- save binary mask instead of gaussian <br/>
- edit comments describing files saved

- update consep trained model to remove extra part
- add link to model
- save model output with visualization 

- add get prediction coordinates

### Generate Ground Truth Labels 
1. Download and unzip the CoNSeP dataset to the directory `../MCSpatNet_datasets`
	

	     wget https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/consep.zip -P ../MCSpatNet_datasets
	     unzip ../datasets/consep.zip -d ../MCSpatNet_datasets

2. `cd data_prepare/`
3. Edit `1_generate_dot_maps_consep.py` <br/>
Set the variables: <br/>
`in_dir` points to the CoNSeP train/test directory, and <br/>
`out_root_dir` points to the training/testing data output directory, respectively. <br/>
Default values are:

	     in_dir = '../../MCSpatNet_datasets/CoNSeP/Train' 
	     out_root_dir = '../../MCSpatNet_datasets/CoNSeP_train' 
         
4. Run `1_generate_dot_maps_consep.py`


		python 1_generate_dot_maps_consep.py

	It will create 2 sub-directories: `images` and `gt_custom` in the output folder.<br/>
	The generated files are:	<br/>		

	- images/: <br/>
		- `<img_name>.png`: the rescaled images by 0.5 (20x). <br/>
	- gt\_custom/: <br/>
		- `<img_name>_gt_dots.npy`: the classification dot annotation map. <br/>
		- `<img_name>_gt_dots_all.npy`: the detection dot annotation map. <br/>
		- `<img_name>.npy`: the classification binary mask. <br/>
		- `<img_name>_all.npy`: the detection binary mask. <br/>
		- `<img_name>_s<class id>_binary.png`: visualization of the binary mask for each class (default: 1=inflammatory, 2=epithelial, 3=stromal).<br/>
		- `<img_name>_binary.png`: visualization of the detection binary mask.<br/>
		- `<img_name>_img_with_dots.jpg`: image with cells dot annotation visualization with different dot colors. (default: blue=inflammatory, red=epithelial, green=stromal).<br/>
			
5. Edit `2_calc_kmaps.py` <br/>
Set the variables: <br/>
`root_dir` points to the CoNSeP train/test directory created in the previous step <br/>
Default value is:

	     root_dir = '../../MCSpatNet_datasets/CoNSeP_train' 

6. Run `2_calc_kmaps.py`

		python 2_calc_kmaps.py

	It will create the sub-directory: `k_func_maps` in the output folder.<br/>
	It generates the cross k function maps. The file names are `k_func_maps/<img_name>_gt_kmap.npy` <br/>		

7. Repeat steps 3-6 with the test data directory:<br/>
	Replace `CoNSeP/Train` with `CoNSeP/Test` <br/>
	Replace `CoNSeP_train` with `CoNSeP_test`
	
	
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
