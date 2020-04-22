# StrainNet (Pytorch implementation)

StrainNet [1] estimates subpixelic displacement and strain fields from pairs of reference and deformed images of a flat speckled surface, as Digital Image Correlation (DIC) does. It is based on FlowNetS [2]. See paper [1] for details. 

Please cite the papers [1] and [2]. Also, make sure to adhere to the licensing terms of the authors. 


## Prerequisite

Install the following modules: 
	
	pytorch >= 1.2
	torchvision
	tensorboardX 
	imageio
	argparse
	path.py
	numpy
	pandas
       
## Training

1. Generate Speckle dataset [1.0](https://github.com/seyfeddineboukhtache/StrainNet/tree/master/Dataset/Speckle%20dataset) or [2.0](https://github.com/seyfeddineboukhtache/StrainNet/tree/master/Dataset/Speckle%20dataset%202.0)

2. Specify the dataset path in Train.py

3. Execute the following command 

       python Train.py --arch StrainNet_h
   
       python Train.py --arch StrainNet_f


## Running inference

       python inference.py /path/to/input/images/  --arch StrainNet_h  --pretrained /path/to/pretrained/model

       python inference.py /path/to/input/images/  --arch StrainNet_f  --pretrained /path/to/pretrained/model  

## Pretrained Models

The pretrained models of StrainNet-h and -f are available [here](https://drive.google.com/drive/folders/1eh2h6ysikk87L_uad8NNt4FpEq7BSN9M?usp=sharing)

## Results of star images

|Reference |<img src="https://github.com/seyfeddineboukhtache/StrainNet/blob/master/Star_frames/Displacements/Reference.png" width="700" height="150">  |
| ---------|-------------------------------------------------------------------------------------------------------------------------|
|StrainNet-h |<img src="https://github.com/seyfeddineboukhtache/StrainNet/blob/master/Star_frames/Displacements/StrainNet-h.png" width="700" height="150">|
|StrainNet-f |<img src="https://github.com/seyfeddineboukhtache/StrainNet/blob/master/Star_frames/Displacements/StrainNet-f.png" width="700" height="150">|


## References 
[1]


[2] A. Dosovitskiy et al., "FlowNet: Learning Optical Flow with Convolutional Networks," 2015 IEEE International Conference on Computer Vision (ICCV), Santiago, 2015, pp. 2758-2766.


