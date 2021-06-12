import torch.nn as nn


class CNN_Block_3D(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(CNN_Block_3D, self).__init__()
        self.conv = nn.Conv3d(in_chn, out_chn, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.bn = nn.BatchNorm3d(out_chn)
    
    def forward(self, x):
        return self.bn(self.maxpool(self.relu(self.conv(x))))
        

class CNN_Extractor_3D(nn.Module):
    def __init__(self):
        super(CNN_Extractor_3D, self).__init__()
        self.conv_block1 = CNN_Block_3D(in_chn=1, out_chn=32)
        self.conv_block2 = CNN_Block_3D(in_chn=32, out_chn=64)
        self.conv_block3 = CNN_Block_3D(in_chn=64, out_chn=128)
    
    def forward(self, x):
        return self.conv_block3(self.conv_block2(self.conv_block1(x)))


class CNN_Block_2D(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(CNN_Block_2D, self).__init__()
        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        self.bn = nn.BatchNorm2d(out_chn)
    
    def forward(self, x):
        return self.bn(self.maxpool(self.relu(self.conv(x))))
        

class CNN_Extractor_2D(nn.Module):
    def __init__(self):
        super(CNN_Extractor_2D, self).__init__()
        self.conv_block1 = CNN_Block_2D(in_chn=25, out_chn=32)
        self.conv_block2 = CNN_Block_2D(in_chn=32, out_chn=64)
        self.conv_block3 = CNN_Block_2D(in_chn=64, out_chn=128)
    
    def forward(self, x):
        return self.conv_block3(self.conv_block2(self.conv_block1(x)))
