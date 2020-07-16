import argparse
from path import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import models
from tqdm import tqdm
import time
import torchvision.transforms as transforms
from imageio import imread, imwrite
import numpy as np


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='StrainNet inference',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--arch', default='StrainNet_f',choices=['StrainNet_f','StrainNet_h'],
                    help='network f or h')                                  
parser.add_argument('data', metavar='DIR',
                    help='path to images folder, image names must match \'[name]0.[ext]\' and \'[name]1.[ext]\'')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained model')
parser.add_argument('--output', '-o', metavar='DIR', default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--div-flow', default=2, type=float,
                    help='value by which flow will be divided')
parser.add_argument("--img-exts", metavar='EXT', default=['tif','png', 'jpg', 'bmp', 'ppm'], nargs='*', type=str,
                    help="images extensions to glob")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()

    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))
    if args.output is None:
        save_path = data_dir/'flow'
    else:
        save_path = Path(args.output)
    print('=> will save everything to {}'.format(save_path))
    save_path.makedirs_p()
    
    # Data loading code
    input_transform = transforms.Compose([transforms.Normalize(mean=[0,0,0], std=[255,255,255])
    ])

    img_pairs = []
    for ext in args.img_exts:
        test_files = data_dir.files('*1.{}'.format(ext))
        for file in test_files:
            img_pair = file.parent / (file.namebase[:-1] + '2.{}'.format(ext))
            if img_pair.isfile():
                img_pairs.append([file, img_pair])

    print('{} samples found'.format(len(img_pairs)))
    
    # create model
    network_data = torch.load(args.pretrained)
    print("=> using pre-trained model '{}'".format(args.arch))
    model = models.__dict__[args.arch](network_data).to(device)
    model.eval()
    cudnn.benchmark = True

    for (img1_file, img2_file) in tqdm(img_pairs):
    
        img1 =  np.array(imread(img1_file))
        img2 =  np.array(imread(img2_file))
        img1 = img1/255
        img2 = img2/255
        		
        if img1.ndim == 2:         
            img1 = img1[np.newaxis, ...]       
            img2 = img2[np.newaxis, ...]
        
            img1 = img1[np.newaxis, ...]       
            img2 = img2[np.newaxis, ...]
            
            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()       
        
            in_ref = torch.cat([img1,img1,img1],1)
            in_def = torch.cat([img2,img2,img2],1)
            input_var = torch.cat([in_ref,in_def],1)           

        elif img1.ndim == 3:
            img1 = np.transpose(img1, (2, 0, 1))
            img2 = np.transpose(img2, (2, 0, 1))        
        
            img1 = torch.from_numpy(img1).float()
            img2 = torch.from_numpy(img2).float()       
            input_var = torch.cat([img1, img2]).unsqueeze(0)          
        
        # compute output
        input_var = input_var.to(device)
        output = model(input_var)
        if args.arch == 'StrainNet_h':
            output = output = torch.nn.functional.interpolate(input=output, scale_factor=2, mode='bilinear')
 
        
        output_to_write = output.data.cpu()
        output_to_write = output_to_write.numpy()       
        disp_x = output_to_write[0,0,:,:]
        disp_x = - disp_x * args.div_flow + 1        
        disp_y = output_to_write[0,1,:,:]
        disp_y = - disp_y * args.div_flow + 1
        
        filenamex = save_path/'{}{}'.format(img1_file.namebase[:-1], '_disp_x')
        filenamey = save_path/'{}{}'.format(img1_file.namebase[:-1], '_disp_y')        
        np.savetxt(filenamex + '.csv', disp_x,delimiter=',')
        np.savetxt(filenamey + '.csv', disp_y,delimiter=',')
        
if __name__ == '__main__':
    main()

