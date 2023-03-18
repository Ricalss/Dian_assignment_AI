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
        #self.input.requires_grad = True #保存input，并且开辟梯度空间，后续尝试是否能去掉。外部已经定义
        # padding 
        pi=self.padding[0]
        pj=self.padding[1]
        if pi == 0 and pj == 0:
            pass
        else:
            _input = torch.zeros(input.size(0),input.size(1),input.size(2)+2*pi,input.size(3)+2*pj)
            for bs in range(input.size(0)):
                for ch in range(input.size(1)):
                    for i in range(input.size(2)):
                        for j in range(input.size(3)):
                            if i-pi>=0 or i < _input.size(2)-pi\
                                or j-pj>=0 or j < _input.size(3)-pj:
                                    _input[bs][ch][i][j]=input[bs][ch][i-pi][j-pj]
            input = _input
        self.inputp =input     #获得padding后的输入张量，为weight和bias的backward准备

        #实现卷积
        dim2 = int((self.input.size(2)+2*padding[0]-kernel.size(2))/stride[0]+1)
        dim3 = int((self.input.size(3)+2*padding[1]-kernel.size(3))/stride[1]+1)
        ks =kernel.size(2)
        self.output = torch.zeros(input.size(0), self.out_channels,dim2,dim3)
        for X0 in range(input.size(0)):
            for X1 in range(kernel.size(0)):
                for X2 in range(dim2):
                    for X3 in range(dim3):
                        self.output[X0][X1][X2][X3]=(input[X0][:,X2:X2+ks,X3:X3+ks]*kernel[X1]).sum().item()+bias[X1]

        #------------------------------------------------------------------------------------------------------------------
        return self.output
    
    def forward(self, input: Tensor):
        weight = self.weight
        bias = self.bias
        padding =self.padding
        stride =self.stride
        return self.conv2d(input, weight, bias, stride, padding)
    
    def backward(self, ones: Tensor):
        '''TODO backward的计算方法''' 
        #-------------------------------------------------------------------------------------------------------------------
        #ones  self   self.weight  self.bias
        #在forward中储存input数据为self.input,因为反向传播计算weight的梯度需要
        #bias.backward
        #self.bias.requires_grad = True
        self.bias.grad=torch.zeros(self.bias.size(0))
        for i in range(self.bias.size(0)):#
            _sum =0.
            for bs in range(ones.size(0)):
                for j in range(ones.size(2)):
                    for k in range(ones.size(3)):
                        _sum += ones[bs][i][j][k]
            self.bias.grad[i] = _sum
        #weight.backward
        #self.weight.requires_grad = True
        self.weight.grad=torch.zeros(self.weight.size(0),self.weight.size(1),self.weight.size(2),self.weight.size(3))
        idim2 = ones.size(2)
        idim3 = ones.size(3)
        for kn in range(self.weight.size(0)):
            for ch in range(self.weight.size(1)):
                for i in range(self.weight.size(2)):
                    for j in range(self.weight.size(3)):#self.inputp是padding后的tensor
                        #self.weight.grad[kn][ch][i][j] =sum(\
                         #   self.inputp[bs][ch][ii*self.stride[0]+i][jj*self.stride[1]+j]*ones[bs][kn][ii][jj]\
                          #      for bs in range(ones.size(0))for ii in range(idim2) for jj in range(idim3))
                        _sum =0.
                        for bs in range(ones.size(0)):
                            for ii in range(idim2):
                                for jj in range(idim3):
                                    _sum +=self.inputp[bs][ch][ii*self.stride[0]+i][jj*self.stride[1]+j]*ones[bs][kn][ii][jj]
                        self.weight.grad[kn][ch][i][j] = _sum
    
        #input.backward to be finished
        #self.input.requires_grad =True
        self.input.grad = torch.zeros(self.input.size(0),self.input.size(1),self.input.size(2),self.input.size(3))
        for bs in range(self.input.size(0)):
            for ch in range(self.input.size(1)):
                for i in range(self.input.size(2)):
                    for j in range(self.input.size(3)):
                        sum2=0.
                        for kn in range(self.weight.size(0)): # ch is kowned
                            for ii in range(self.weight.size(2)):
                                for jj in range(self.weight.size(3)):
                                    if i-ii+self.padding[0]<0 \
                                        or i-ii>self.input.size(2)+self.padding[0]-self.weight.size(2) \
                                        or j-jj+self.padding[1]<0 \
                                        or j-jj>self.input.size(3)+self.padding[1]-self.weight.size(3) :
                                        sum2 +=0.
                                    else:
                                        sum2 += self.weight[kn][ch][ii][jj]*ones[bs][kn][i-ii+self.padding[0]][j-jj+self.padding[1]]
                        self.input.grad[bs][ch][i][j] =sum2
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
        self.input = input
        #self.input.requires_grad = True #保存input，并且开辟梯度空间
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
        #-------------------------------------------------------------------------------------------------------------------
        #bias.backward
        #self.bias.requires_grad = True
        self.bias.grad=torch.zeros(self.bias.size(0))
        for i in range(self.bias.size(0)):
            self.bias.grad[i]=sum(ones[bs][i]for bs in range(ones.size(0)))
        #weight.backward
        #self.weight.requires_grad = True
        self.weight.grad=torch.zeros(self.weight.size(0),self.weight.size(1))
        for out in range(self.weight.size(0)):
            for i in range(self.weight.size(1)):
                self.weight.grad[out][i]=sum(self.input[bs][i]*ones[bs][out] for bs in range(self.input.size(0)))
        #input.backward
        #self.input.requires_grad = True
        self.input.grad = torch.zeros(self.input.size(0),self.input.size(1))
        for bs in range(self.input.size(0)):
            for i in range(self.input.size(1)):
                self.input.grad[bs][i]=sum(self.weight[out][i]*ones[bs][out] for out in range(self.weight.size(0)))
        
        #-------------------------------------------------------------------------------------------------------------------
        return self.input.grad

class CrossEntropyLoss():
    def __init__(self):
        pass
    def __call__(self, input, target):
        '''TODO'''
        #-------------------------------------------------------------------------------------------------
        self.input = input
        #self.input.requires_grad = True #保存input，并且开辟梯度空间
        self.target = target#保存target
        #NLLloss+log+softmax
        batch_size = target.size(0)
        #softmax
        _input = torch.zeros((input.size(0),input.size(1)))
        for i in range(input.size(0)):
            sum_p =sum(torch.exp(input[i][z])for z in range(input.size(1)))
            for j in range(input.size(1)):
                _input[i] = (torch.exp(input[i])) /sum_p
                
        self.SMoutput = _input   #保存数据用于后续backward计算，是为一维
        
        #log()
        _input =log(_input)
        #NLLloss
        self.output = 0.
        for j in range(batch_size):
            self.output += -_input[j][target[j]]
        self.output =self.output / batch_size
        #---------------------------------------------------------------------------------------------------
        return self.output
    def backward(self):
        '''TODO'''
        #--------------------------------------------------------------------------------------------------
        self.input.grad=torch.zeros_like(self.input)
        #NLLloss+log().backward
        _input = torch.zeros(self.input.size(0))
        for i in range(self.input.size(0)):
            for j in range (self.input.size(1)): 
                if j ==self.target[i]:
                    _input[i] = (1/self.SMoutput[i][j])*(-1/self.input.size(0))#SMoutput是二维，但是_output是一维
        #softmax.backward
        for bs in range(self.input.size(0)):
            mol1 =torch.exp(self.input[bs][self.target[bs]]).item()#分子1
            den =torch.exp(self.input[bs]).sum().item()#分母
            for i in range(self.input.size(1)):
                mol2 =torch.exp(self.input[bs][i]).item()#分子2
                if i==self.target[bs]:
                    self.input.grad[bs][i]=(den-mol1)*mol1/(den**2)*_input[bs]
                else:
                    self.input.grad[bs][i]=(-(mol1*mol2)/(den**2))*_input[bs]
                
        
        #---------------------------------------------------------------------------------------------------
        return self.input.grad
    #请移步CNN-self分支
