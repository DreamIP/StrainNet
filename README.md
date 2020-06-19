# StrainNet (Pytorch implementation)

StrainNet estimates subpixelic displacement and strain fields from pairs of reference and deformed images of a flat speckled surface, as Digital Image Correlation (DIC) does. See paper [1] for details. 

If you find this implementation useful, please cite the papers [1]. Also, make sure to adhere to the licensing terms of the authors. 

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
	tqdm
	
       
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

The pretrained models of StrainNet-h and StrainNet_f are available [here](https://drive.google.com/drive/folders/1eh2h6ysikk87L_uad8NNt4FpEq7BSN9M?usp=sharing) 

## Results of star images

|Reference   | ![](Star_frames/Displacements/Displacements/Reference.png)  |
|------------|-------------------------------------------------------------|
|StrainNet-h | ![](Star_frames/Displacements/Displacements/StrainNet-h.png)|
|StrainNet-f | ![](Star_frames/Displacements/Displacements/StrainNet-f.png)|


## References 
[1] S. Boukhtache, K. Abdelouahab, F. Berry, B. Blaysat, M. Grédiac, F. Sur. When Deep Learning Meets Digital Image Correlation. 2020. Submitted. 

## Acknowledgments

This code is based on the Pytorch impelmentation of FlowNetS from [https://github.com/ClementPinard/FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)

