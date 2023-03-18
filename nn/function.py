from typing import Union
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch
from torch.nn.modules.conv import _ConvNd
# from cnnbase import ConvBase
from torch.nn.modules.utils import _pair
from torch.nn.common_types import _size_2_t
import time#计时功能
class Conv2d(_ConvNd):
   
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        
    def conv2d(self, input, kernel, bias = 0, stride=1, padding=0):
        '''TODO forword的计算方法''' 
        #-----------------------------------------------------------------------------------------------------------------------------
        #padding to be finished
        self.input_padding = input
        
        #kernel computation
        dim2 = int((self.input.size(2)+2*padding[0]-kernel.size(2))/stride[0]+1)
        dim3 = int((self.input.size(3)+2*padding[1]-kernel.size(3))/stride[1]+1)
        ks =kernel.size(2)
        self.output = torch.zeros(input.size(0), self.out_channels,dim2,dim3)
        for X0 in range(input.size(0)):
            for X1 in range(kernel.size(0)):
                for X2 in range(dim2):
                    for X3 in range(dim3):
                        self.output[X0][X1][X2][X3]=(input[X0][:,X2:X2+ks,X3:X3+ks]*kernel[X1]).sum().item()+bias[X1]
        
        #-----------------------------------------------------------------------------------------------------------------------------
        return self.output
    
    def forward(self, input: Tensor):
        weight = self.weight
        bias = self.bias
        return self.conv2d(input, weight, bias)
    
    def backward(self, ones: Tensor):
        '''TODO backward的计算方法''' 
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
        
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))#随机weight(outp, inp)
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
            
            
    def forward(self, input):
        '''TODO'''
        #input (bs,inp)  self.weight(inp,outp)
        #------------------------------------------------------------------------------------------------------
        self.input = input
        self.output = torch.mm(input,self.weight.T)+self.bias
        #------------------------------------------------------------------------------------------------------
        return self.output
    def backward(self, ones: Tensor):
        '''TODO'''
        time_start = time.time()
        #input (bs,inp)  self.weight(inp,outp)  ones = self.output(bs,outp)
        #------------------------------------------------------------------------------------------------------
        # #bias.backweard
        self.bias.grad=torch.sum(ones,dim=0)
        #input.backward
        self.input.grad = torch.mm(ones, self.weight)
        #weight.backward
        self.weight.grad = torch.mm(self.input.T, ones ).T
        #------------------------------------------------------------------------------------------------------
        time_end = time.time()
        print(time_end - time_start)
        return self.input.grad

class CrossEntropyLoss():
    def __init__(self):
        pass
    def __call__(self, input, target):
        '''TODO'''
        #------------------------------------------------------------------------------------------------------
        self.input = input
        self.target = target
        self.tars = self.input.size(0)
        self.Bs = input.size(0)
        #softmax  input(bs, labs) ---> output1(bs, labs)
        self.out_exp = torch.exp(input) #空间换时间
        self.exp_sum = torch.sum(self.out_exp, dim=1).reshape(self.Bs, 1)
        self.label_exp = torch.zeros(self.Bs, 1) #标签值
        for bs in range(self.Bs):
            self.label_exp[bs][0] = self.out_exp[bs][target[bs]]
        self.label_P = self.label_exp / self.exp_sum 
        #log -1 NLLloss output1(bs,labs) ---> self.output = scalar
        self.output = -sum(torch.log(self.label_P)) / self.Bs  
        #------------------------------------------------------------------------------------------------------
        return self.output
    def backward(self):
        '''TODO'''
        time_start = time.time()
        #------------------------------------------------------------------------------------------------------
        vec_bs = -1/self.Bs/self.label_P
        self.input.grad = -self.out_exp*self.label_exp/(self.exp_sum**2) 
        change = (self.label_exp-self.exp_sum)/self.label_exp
        for bs in range(self.Bs):
            self.input.grad[bs][self.target[bs]] *= change[bs][0]
        self.input.grad = self.input.grad*vec_bs
        #------------------------------------------------------------------------------------------------------
        time_end = time.time()
        print(time_end - time_start)
        return self.input.grad