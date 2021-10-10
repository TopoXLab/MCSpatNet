### Environment Set Up 

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

