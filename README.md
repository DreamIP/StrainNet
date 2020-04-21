# StrainNet (Pytorch implementation)

StrainNet estimates subpixelic displacement and strain fields from pairs of reference and deformed images of a flat speckled surface, as Digital Image Correlation (DIC) does. It is based on FlowNetS [1]. See paper [2] for details. 

Please cite the papers [1] and [2], also, make sure to adhere to the licensing terms of the authors. 


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

3. Execute the follwing command 

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
|DIC 21x21 |<img src="https://github.com/seyfeddineboukhtache/StrainNet/tree/master/Star_frames/Displacements/DIC21x21.png" width="700" height="150">|
|DIC 11x11 |<img src="https://github.com/seyfeddineboukhtache/StrainNet/tree/master/Star_frames/Displacements/DIC11x11.png" width="700" height="150">|
|StrainNet-h |<img src="https://github.com/seyfeddineboukhtache/StrainNet/tree/master/Star_frames/Displacements/StrainNet-h.png" width="700" height="150">|
|StrainNet-f |<img src="https://github.com/seyfeddineboukhtache/StrainNet/tree/master/Star_frames/Displacements/StrainNet-f.png" width="700" height="150">|


## References 

[1]
@InProceedings{DFIB15,
  author       = "A. Dosovitskiy and P. Fischer and E. Ilg and P. H{\"a}usser and C. Haz{\i}rba{\c{s}} and V. Golkov and P. v.d. Smagt and D. Cremers and T. Brox",
  title        = "FlowNet: Learning Optical Flow with Convolutional Networks",
  booktitle    = "IEEE International Conference on Computer Vision (ICCV)",
  year         = "2015"
}

[2]
