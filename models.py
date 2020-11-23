import torchvision.models as models
from torch.nn import Parameter
from util import *
import torch
import torch.nn as nn
import numpy as np
import sys
import torch.nn.functional as F

DEBUG_MODEL = False

grads = {}

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


class GraphConvolution(nn.Module):
    
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.Linear_predict = nn.Linear(1000, 3000)

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):

        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNResnet(nn.Module):
    def __init__(self, option, model, num_classes, in_channel=300, t=0, adj_file=None):
        super(GCNResnet, self).__init__()
        self.adj_file = adj_file
        self.opt = option
        self.state = {}
        self.state['use_gpu'] = torch.cuda.is_available()
        self.is_usemfb = option.IS_USE_MFB
        self.pooling_stride = option.pooling_stride
        self.features = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )
        self.d_e = in_channel
        self.d_e_1 = self.opt.inter_channel
        self.A_branch_1 = nn.Sequential(
            nn.Conv2d(self.d_e, self.d_e_1, kernel_size=1), 
            nn.Sigmoid(),
        )
        self.A_branch_2 = nn.Sequential(
            nn.Conv2d(self.d_e, self.d_e_1, kernel_size=1),  
            nn.Sigmoid(),
        )
        self.num_classes = num_classes
        self.pooling = nn.MaxPool2d(14, 14)

        self.gc1 = GraphConvolution(in_channel, 1024)
        self.gc2 = GraphConvolution(1024, 2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = adj2tensor(adj_file)
        
        self.A = Parameter(torch.from_numpy(_adj).float())  
        self.image_normalization_mean = [0.485, 0.456, 0.406]     
        self.image_normalization_std = [0.229, 0.224, 0.225]       

        self.JOINT_EMB_SIZE=option.linear_intermediate

        if self.is_usemfb:
            assert self.JOINT_EMB_SIZE % self.pooling_stride == 0, \
                'linear-intermediate value must can be divided exactly by sum pooling stride value!'
            self.out_in_tmp = int(self.JOINT_EMB_SIZE / self.pooling_stride)
            self.ML_fc_layer = nn.Linear(int(self.num_classes * self.out_in_tmp), int(self.num_classes))
        else:
            self.out_in_tmp = int(1)
        
        self.Linear_imgdataproj = nn.Linear(option.IMAGE_CHANNEL, self.JOINT_EMB_SIZE)  
        self.Linear_classifierproj = nn.Linear(option.CLASSIFIER_CHANNEL, self.JOINT_EMB_SIZE)  

    def forward(self, feature, inp):
        feature = self.features(feature)
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)               
        inp = inp[0]              
        
        branch_1 = self.A_branch_1(inp.view((1,self.d_e, self.num_classes, 1))).view((self.d_e_1, self.num_classes))
        branch_2 = self.A_branch_2(inp.view((1,self.d_e, self.num_classes, 1))).view((self.d_e_1, self.num_classes))
        A = torch.matmul(branch_1.t(), branch_2)/float(self.num_classes)
        
        I_c = torch.eye(A.shape[0]).cuda() if torch.cuda.is_available() else torch.eye(A.shape[0]).cpu()
        A_wave = A + I_c
        D_wave_negative_power = torch.diag(torch.pow(A_wave.sum(1).float(),-0.5))
        D_wave_negative_power[torch.isnan(D_wave_negative_power)] = 0.0
        D_wave_negative_power[torch.isinf(D_wave_negative_power)] = 0.0
        A_hat = torch.matmul(torch.matmul(D_wave_negative_power, A_wave), D_wave_negative_power)
        L_A_loss = torch.abs(A_hat - I_c).sum()
        if L_A_loss!=L_A_loss:
            print("A = \n", A)
            print("A_wave = \n", A_wave)
            print("A_wave sum(1) = \n", A_wave.sum(1))
            print("D~ diagnose elements = \n", torch.pow(A_wave.sum(1).float(),-0.5))
            print("D~ = \n", D_wave_negative_power)
            print("A_hat=\n", A_hat)
            sys.exit()
        
        adj = A_hat
        
        x = self.gc1(inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        if self.state['use_gpu']:
            x_out = torch.FloatTensor(torch.FloatStorage()).cuda()
        else:
            x_out = torch.FloatTensor(torch.FloatStorage())
        if self.is_usemfb:
            for i_row in range(int(feature.shape[0])):
                img_linear_row = self.Linear_imgdataproj(feature[i_row, :]).view(1, -1)
                if self.state['use_gpu']:
                    out_row = torch.FloatTensor(torch.FloatStorage()).cuda()
                else:
                    out_row = torch.FloatTensor(torch.FloatStorage())
                for col in range(int(x.shape[1])): 
                    tmp_x = x[:, col].view(1, -1)  
                    classifier_linear = self.Linear_classifierproj(tmp_x)  
                    iq = torch.mul(img_linear_row, classifier_linear)  
                    iq = F.dropout(iq, self.opt.DROPOUT_RATIO, training=self.training)  
                    iq = torch.sum(iq.view(1, self.out_in_tmp, -1), 2) 
                    out_row = torch.cat((out_row, iq), 1)
                
                if self.out_in_tmp != 1: 
                    temp_out = self.ML_fc_layer(out_row)
                    out_row = temp_out  
        
                x_out = torch.cat((x_out, out_row), 0)  

        else:   x_out = torch.matmul(feature, x)      
        assert x_out.shape[0]==feature.shape[0]
        
        return x_out, L_A_loss

    def get_config_optim(self, lr, lrp):
        return [
                    
                    {'params': self.features.parameters(), 'lr': lr * lrp},
                    {'params': self.A_branch_1.parameters(), 'lr': 4.0 * lr},
                    {'params': self.A_branch_2.parameters(), 'lr': 4.0 * lr},
                    
                    {'params': self.gc1.parameters(), 'lr': lr},
                    {'params': self.gc2.parameters(), 'lr': lr},
                ]

def gcn_resnet101(opt, num_classes, t, pretrained=True, adj_file=None, in_channel=300):
    
    model = models.resnet101(pretrained=pretrained)
    
    return GCNResnet(opt, model, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
