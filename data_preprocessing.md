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
	
