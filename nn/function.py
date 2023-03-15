from typing import Union
from certifi import where
from numpy import arange
from torch import Tensor, log, rand
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch
from torch.nn.modules.conv import _ConvNd
# from cnnbase import ConvBase
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t

class Conv2d(_ConvNd):
    #此处初始化会带入weight，bias，初始化情况由传入参数决定
    def __init__(                        
        self,
        in_channels: int, #C:3
        out_channels: int,#Cp:5
        kernel_size: _size_2_t,#k_s:2
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type "zeros"表示边缘填充0
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size) #打包为元组，因为卷积核尺度有宽和高
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        
    def conv2d(self, input, kernel, bias = 0, stride=1, padding=0):
        '''TODO forword的计算方法''' 
        #-----------------------------------------------------------------------------------------------------------------
        self.input =input
        self.input.requires_grad = True #保存input，并且开辟梯度空间，后续尝试是否能去掉。外部已经定义
        # padding to be finished
        
        
        self.inputp =input#获得padding后的输入张量，为weight和bias的backward准备

        #实现卷积
        dim2 = int((input.size(2)+2*padding-kernel.size(2))/stride+1)
        dim3 = int((input.size(3)+2*padding-kernel.size(3))/stride+1)
        ks =kernel.size(2)
        self.output = torch.zeros(input.size(0), self.out_channels,dim2,dim3)
        
        for X0 in range(input.size(0)):
            for X1 in range(kernel.size(0)):
                for X2 in range(dim2):
                    for X3 in range(dim3):
                        self.output[X0][X1][X2][X3]=(input[X0][:,X2:X2+ks,X3:X3+ks]*kernel[X1]).sum()+bias[X1]
        print(self.input.grad)
        #------------------------------------------------------------------------------------------------------------------
        return self.output
    
    def forward(self, input: Tensor):
        weight = self.weight
        bias = self.bias
        return self.conv2d(input, weight, bias)
    
    def backward(self, ones: Tensor):
        '''TODO backward的计算方法''' 
        #-------------------------------------------------------------------------------------------------------------------
        #ones  self   self.weight  self.bias
        #在forward中储存input数据为self.input,因为反向传播计算weight的梯度需要
        for i in range(self.bias.size(0)):#
            _sum =0.
            self.bias.grad[i] = _sum
            for bs in ones.size(0):
                for j in range(ones.size(2)):
                    for k in range(ones.size(3)):
                        _sum += ones[bs][i][j][k]
        #weight.backward
        dim2 = int((input.size(2)-self.weight.size(2))/self.stride+1)
        dim3 = int((input.size(2)-self.weight.size(3))/self.stride+1)
        for bs in range(ones.size(0)):
            for kn in range(self.weight.size(0)):
                for ch in range(self.weight.size(1)):
                    for i in range(self.weight.size(2)):
                        for j in range(self.weight.size(3)):
                            self.weight.grad[kn][ch][i][j] =sum(\
                                self.input[bs][ch][i][j]*ones[bs][kn][ii][jj]\
                                    for ii in range(0,dim2,self.stride) for jj in range(0,dim3,self.stride))
    
        #input.backward to be finished
        
        #-------------------------------------------------------------------------------------------------------------------
        return self.input.grad
    
class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))#随机weight
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))# 返回未初始化数据的张量
            
            
    def forward(self, input):
        '''TODO'''
        #-----------------------------------------------------------------------------------------
        self.output = torch.zeros(input.size(0),self.weight.size(0))
        for i in range(input.size(0)):
            for j in range(self.weight.size(0)):
                self.output[i][j] = sum((self.weight[j][k]*input[i][k]for k in range(self.weight.size(1))))
    
        #添加bias值
        if type(self.bias) ==bool :
            pass
        else:
            for i in arange(self.output.size(0)):
                self.output[i] =self.output[i] + self.bias
            
        #-----------------------------------------------------------------------------------------
        return self.output
    def backward(self, ones: Tensor):
        '''TODO'''
        return self.input.grad

class CrossEntropyLoss():
    def __init__(self):
        pass
    def __call__(self, input, target):
        '''TODO'''
        #---------------------------------------------------------------------
        #NLLloss+log+softmax
        batch_size = target.size(0)
        #softmax
        _input = torch.zeros((input.size(0),input.size(1)))
        for i in range(input.size(0)):
            sum_p =sum(torch.exp(input[i][z])for z in range(input.size(1)))
            for j in range(input.size(1)):
                _input[i] = (torch.exp(input[i])) /sum_p
        #log()
        _input =log(_input)
        #NLLloss
        self.output = 0.
        for j in range(batch_size):
            self.output += -_input[j][target[j]]
        self.output =self.output / batch_size
        #---------------------------------------------------------------------
        return self.output
    def backward(self):
        '''TODO'''
        return self.input.grad
        
