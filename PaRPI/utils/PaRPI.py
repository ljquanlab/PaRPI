import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
import numpy as np
from math import log
from utils.conv_layer import *
from utils.cross_attention import *
from utils.graphsage_layer import *
from einops.layers.torch import Rearrange
import dgl
import dgl.function as fn
import dgl.ops as ops
from utils.mlp_readout_layer import MLPReadout


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionResNetBlock(nn.Module):
    def __init__(self, in_channels, scale=0.5):
        super(InceptionResNetBlock, self).__init__()
        self.scale = scale
        
        self.branch1x1 = BasicConv1d(in_channels, 64, kernel_size=1)

        self.branch3x3_1 = BasicConv1d(in_channels, 48, kernel_size=1)
        self.branch3x3_2 = BasicConv1d(48, 64, kernel_size=3, padding=1)

        self.branch5x5_1 = BasicConv1d(in_channels, 64, kernel_size=1)
        self.branch5x5_2 = BasicConv1d(64, 96, kernel_size=5, padding=2)

        self.conv2d = nn.Conv1d(224, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        
        outputs = [branch1x1, branch3x3, branch5x5]
        concatenated = torch.cat(outputs, 1)

        upsampled = self.conv2d(concatenated)
        upsampled = self.bn(upsampled)

        if self.scale != 1:  
            x = x * self.scale
        x = x + upsampled
        x = self.relu(x)
        return x

class CoordAttention(nn.Module):
    def __init__(self, channel, reduction=32):
        super(CoordAttention, self).__init__()
        self.conv1 = nn.Conv1d(channel, channel // reduction, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channel // reduction, channel, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_pool = F.adaptive_avg_pool1d(x, 1)
        x = self.conv1(x_pool)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x * x_pool


class CBAMBlock(nn.Module):
    def __init__(self, channel):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channel, channel // 8, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channel // 8, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_out = self.channel_attention(x) * x
        return x_out


class transformer(nn.Module):
    def __init__(self, input_channels):
        super(transformer, self).__init__()
        self.inception_resnet_block = InceptionResNetBlock(input_channels)
        self.cbam = CBAMBlock(input_channels)
        self.coord_attention = CoordAttention(input_channels)
        self.shuffle_transformer = nn.TransformerEncoderLayer(d_model=input_channels, nhead=8)
        self.classifier = nn.Linear(input_channels, 1)
        
    def forward(self, x):
        x = self.inception_resnet_block(x)
        x = self.cbam(x)
        x = x.transpose(1, 2)
        x = self.shuffle_transformer(x)
        x = x.transpose(1, 2)
        return x



class DPRBPblock(nn.Module):
    def __init__(self, filter_num, kernel_size, dilation):
        super(DPRBPblock, self).__init__()
        self.conv = Conv1d(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation,
                           same_padding=False)
        self.conv1 = Conv1d(filter_num, filter_num, kernel_size=kernel_size, stride=1, dilation=dilation,
                            same_padding=False)
        self.max_pooling = nn.MaxPool1d(kernel_size=(3,), stride=2)
        self.padding_conv = nn.ConstantPad1d(((kernel_size - 1) // 2) * dilation, 0)
        self.padding_pool = nn.ConstantPad1d((0, 1), 0)

    def forward(self, x):
        x = self.padding_pool(x)
        px = self.max_pooling(x)
        x = self.padding_conv(px)
        x = self.conv(x)
        x = self.padding_conv(x)
        x = self.conv1(x)
        x = x + px

        return x


class DPRBP(nn.Module):

    def __init__(self, filter_num, number_of_layers):
        super(DPRBP, self).__init__()

        self.kernel_size_list = [1 + x * 2 for x in range(number_of_layers)]
        self.dilation_list = [1, 1, 1, 1, 1, 1]
        self.conv = Conv1d(filter_num, filter_num, self.kernel_size_list[0], stride=1, dilation=1, same_padding=False)
        self.conv1 = Conv1d(filter_num, filter_num, self.kernel_size_list[0], stride=1, dilation=1, same_padding=False)
        self.pooling = nn.MaxPool1d(kernel_size=(3,), stride=2)
        self.padding_conv = nn.ConstantPad1d(((self.kernel_size_list[0] - 1) // 2), 0)
        self.padding_pool = nn.ConstantPad1d((0, 1), 0)

        self.DPRBPblocklist = nn.ModuleList(
            [DPRBPblock(filter_num, kernel_size=self.kernel_size_list[i],
                        dilation=self.dilation_list[i]) for i in range(len(self.kernel_size_list))]
        )
        self.classifier = nn.Linear(filter_num, 1)

    def forward(self, x):
        x = self.padding_conv(x)
        x = self.conv(x)
        x = self.padding_conv(x)
        x = self.conv1(x)
        i = 0
        while x.size()[-1] > 2:
            x = self.DPRBPblocklist[i](x)
            i += 1

        return x


class PaRPI(nn.Module):
    def __init__(self, k=3):
        super(PaRPI, self).__init__()
        number_of_layers = int(log(101 - k + 1, 2))
        self.conv0 = Conv1d(768, 128, kernel_size=(1,), stride=1)
        self.conv1 = Conv1d(1, 128, kernel_size=(k,), stride=1, same_padding=False)
        self.conv2 = Conv1d(1280, 256, kernel_size=(1,), stride=1)
        self.layers = nn.ModuleList([GraphSageLayer(256, 256, F.relu,
                                               aggregator_type='pool', dropout=0.4, batch_norm=True, residual=False) for _ in range(4)])
        self.cross_attn = CrossAttention(source_dim=256, target_dim=256, hidden_dim=256).to(device="cuda:0")
        self.DPRBP = DPRBP(64 * 4, number_of_layers)
        self._initialize_weights()     
        self.MLP_layer = MLPReadout(256, 1)
        self.transformer = transformer(input_channels=256)
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        
    def forward(self, g, e, bert_embedding, structure, protein):
        x0 = bert_embedding  
        x1 = structure  
        x0 = self.conv0(x0)  
        x1 = self.conv1(x1)  
        
        x_1 = torch.cat([x0, x1], dim=1)  
        x_1 = x_1.transpose(1, 2)
        x_1 = x_1.reshape(-1, 256) 

        g.ndata['feat'] = x_1
        for conv in self.layers:
            h = conv(g, x_1, e)        
        g.ndata['h'] = h 
        
        batch_size = structure.shape[0] 
        h_reshaped = h.reshape(batch_size, 99, 256)
        h_reshaped = x_1.reshape(batch_size, 99, 256)
        rna_feature = h_reshaped.transpose(1, 2)
       
        rna_feature = self.transformer(rna_feature)
        rna_feature = self.DPRBP(rna_feature)
        rna_feature = rna_feature.transpose(1, 2) 
        
        x2 = protein 
        x2 = self.conv2(x2)
        x2 = x2.transpose(1, 2)
        
        x_2 = self.cross_attn(rna_feature, x2)
        x_2 = x_2.transpose(1, 2)
        x_2 = x_2.squeeze(-1)

        return self.MLP_layer(x_2)



