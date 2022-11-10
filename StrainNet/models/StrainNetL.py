import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from .util import conv, predict_flow, deconv, crop_like

__all__ = [
    'StrainNet_l', 'StrainNet_l_bn'
]


class StrainNetL(nn.Module):
    expansion = 1

    def __init__(self,batchNorm=True):
        super(StrainNetL,self).__init__()

        self.batchNorm = batchNorm
    
        self.conv1   = conv(self.batchNorm,  2,  48, kernel_size=7, stride=1)
        self.conv2   = conv(self.batchNorm, 48,  64, kernel_size=5, stride=2)
        self.conv3   = conv(self.batchNorm, 64,  128, kernel_size=5, stride=2)
        self.conv4   = conv(self.batchNorm, 128,  128, stride=2)

        self.deconv3 = deconv(128,64)
        self.deconv2 = deconv(194,32)

 
        self.predict_flow4 = predict_flow(128)
        self.predict_flow3 = predict_flow(194) 
        self.predict_flow2 = predict_flow(98) 

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2, 2, 4, 2, 1, bias=False)
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
    
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3) 
        
        flow4       = self.predict_flow4(out_conv4)
        flow4_up    = crop_like(self.upsampled_flow4_to_3(flow4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(out_conv4), out_conv3)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        flow3       = self.predict_flow3(concat3)
        flow3_up    = crop_like(self.upsampled_flow3_to_2(flow3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        flow2 = self.predict_flow2(concat2)

        if self.training:
            return flow2,flow3,flow4
        else:
            return flow2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def StrainNet_l(data=None):
 
    model = StrainNetL(batchNorm=False)
    if data is not None:
         model.load_state_dict(data['state_dict'])
    return model


def StrainNet_l_bn(data=None):
    
    model = StrainNetL(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
