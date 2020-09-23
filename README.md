# StrainNet (Pytorch implementation)

StrainNet estimates subpixelic displacement and strain fields from pairs of reference and deformed images of a flat speckled surface, as Digital Image Correlation (DIC) does. See paper [1] for details. 

If you find this implementation useful, please cite reference [1]. Also, make sure to adhere to the licensing terms of the authors. 

## Prerequisite

Install the following modules: 

```
pytorch >= 1.2
torchvision
tensorboardX 
imageio
argparse
path.py
numpy
pandas
tqdm
```

## Training
1. Generate Speckle dataset [1.0](Dataset/Speckle%20dataset%201.0) or [2.0](Dataset/Speckle%20dataset%202.0)
2. Specify the paths to:
    Train dataset, Test dataset, Train_annotations.csv, and Test_annotations.csv in the file Train.py (exactly in the definition of train_set and test_set)
3. Execute the following commands
```
python Train.py --arch StrainNet_h 
python Train.py --arch StrainNet_f
```

## Running inference

The images pairs should be in the same location, with the name pattern *1.ext  *2.ext

```bash
python inference.py /path/to/input/images/  --arch StrainNet_h  --pretrained /path/to/pretrained/model
python inference.py /path/to/input/images/  --arch StrainNet_f  --pretrained /path/to/pretrained/model  
```

## Pretrained Models

The pretrained models of StrainNet-h and StrainNet_f are available [here](https://drive.google.com/drive/folders/1eh2h6ysikk87L_uad8NNt4FpEq7BSN9M?usp=sharing) 

## Results of star images

Execute the following commands in the StrainNet directory (please also copy here the tar files if you use the pretrained models)

```bash
python inference.py ../Star_frames/Noiseless_frames/  --arch StrainNet_h  --pretrained StrainNet-h.pth.tar
python inference.py ../Star_frames/Noiseless_frames/  --arch StrainNet_f  --pretrained StrainNet-f.pth.tar
```
The output of inference.py can be found in Star_frames/Noiseless_frames/flow/

You can use Script_flow.m to visualize the obtained displacements 

|Reference image   | ![](Star_frames/Displacements/Star.png)   |
|:----------:|:---------------------------------------------:|
|Reference displacement   | ![](Star_frames/Displacements/Reference.png)  |
|Retrieved by StrainNet-h  | ![](Star_frames/Displacements/StrainNet-h.png)|
|Retrieved by StrainNet-f | ![](Star_frames/Displacements/StrainNet-f.png)|


## Reference 
[1] S. Boukhtache, K. Abdelouahab, F. Berry, B. Blaysat, M. Grédiac and F. Sur. *"When Deep Learning Meets Digital Image Correlation"*, *Optics and Lasers in Engineering*, Volume 136, 2021. Available at:

https://www.sciencedirect.com/science/article/pii/S0143816620306588?via%3Dihub 

https://hal.archives-ouvertes.fr/hal-02933431 

https://arxiv.org/abs/2009.03993

## Acknowledgments

This code is based on the Pytorch implmentation of FlowNetS from [FlowNetPytorch](https://github.com/ClementPinard/FlowNetPytorch)
