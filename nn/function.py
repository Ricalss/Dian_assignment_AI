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
        #time_start = time.time()
        #-----------------------------------------------------------------------------------------------------------------------------
        
        self.input = input
        pad_n = self.padding[0]
        bs, ch, x, y = (input.size(i)for i in range(4))
        
        #padding 
        if self.padding[0]!=0 or self.padding[1]!=0 :
            #dataload将数据转换为带batch的四维tensor
            input_padding = torch.zeros((bs,ch,x+2*pad_n,y+2*pad_n))
            input_padding[:, :, pad_n:x+pad_n,pad_n:y+pad_n] = input
        else :
            input_padding = input
        
        #参数
        kn = kernel.size(0)#卷积核个数，输出通道数
        ks = kernel.size(2)#卷积核的尺寸
        dim2 = int((input_padding.size(2)-ks)/self.stride[0]+1)#输出的高
        dim3 = int((input_padding.size(3)-ks)/self.stride[1]+1)#输出的宽
        k_elems = ch*ks*ks  #一个卷积核中元素个数
        lens = k_elems*dim2*dim3   #二维矩阵化时，行向量的长度
        
        #kernel降维---->(kn, ch*ks*ks)
        kernel_2d = kernel.reshape(kn, k_elems)  #1,
        #self.kernel_mat = torch.cat(tuple(kernel_2d for _ in range(dim2*dim3)),1)#延长二维矩阵
        kernel_2d = kernel.reshape(kn, 1, k_elems)  #1
        self.kernel_mat = kernel_2d.expand(-1, dim2*dim3, -1).reshape(kn, lens)#效果
        
        #input_padding降维---->(bs, dim2*dim3*ch*ks*ks) 
        self.pad_mat = torch.zeros(bs, lens)
        for i in range(dim2*dim3):
            n3 = i % dim3
            n2 = int(i / dim3)
            self.pad_mat[:,i*k_elems:i*k_elems+k_elems] = input_padding[:,:,n2:n2+ks,n3:n3+ks].reshape(bs,k_elems)
            
        #compute----self.outoutput----->(bs, kn, dim2 ,dim3)
        self.output = torch.zeros(bs, kn, dim2, dim3)
        bias_3dim = self.bias.reshape(kn,1, 1)
        for b_s in range(bs):   #for k_n in range(kn):取消该循环，可以在外定义bias，使得可以广播相加
            mid_output = (self.kernel_mat*self.pad_mat[b_s]).reshape(kn,dim2,dim3,k_elems)   #可以合并
            self.output[b_s,:,:,:] = torch.sum(mid_output,dim=-1)+bias_3dim
            
            
        #self.output = torch.matmul(self.kernel_mat, self.pad_mat.T).T 彻底的错误
        #matmul可广播
        #-----------------------------------------------------------------------------------------------------------------------------
        #print(time.time() - time_start)
        return self.output
    
    def forward(self, input: Tensor):
        weight = self.weight
        bias = self.bias
        return self.conv2d(input, weight, bias)
    
    def backward(self, ones: Tensor):
        '''TODO backward的计算方法''' 
        #time_start = time.time()
        #------------------------------------------------------------------------------------------------------------------------------
        
        #参数
        kn = self.weight.size(0)
        ch = self.weight.size(1)
        ks = self.weight.size(2)
        bs = self.input.size(0)
        x ,y = self.input.size(2), self.input.size(3)
        pad_n =self.padding[0]   #默认为正方形的图片
        dim2 = int((x+2*pad_n-ks)/self.stride[0]+1)
        dim3 = int((y+2*pad_n-ks)/self.stride[1]+1)
        k_elems = ch*ks*ks
        lens = k_elems*dim2*dim3
        
        
        #bias.backward()
        self.bias.grad = torch.zeros_like(self.bias)
        #for k_n in range(kn):
           # self.bias.grad[k_n] = torch.sum(ones[:,k_n,:,:])  
        self.bias.grad = torch.sum(ones,dim=[0, 2, 3])
        
        #ones三维化
        ones_3dim = torch.cat(tuple(ones for _ in range(k_elems)),3).reshape(bs, kn, lens)
        
        #weight.backward()    weight就是kernel
        #self.weight.grad = torch.zeros_like(self.weight)
        self.weight.grad = torch.sum((ones_3dim*self.pad_mat.reshape(bs, 1, lens)).reshape(bs, kn, dim2*dim3, k_elems), dim=[0,2]).reshape(kn,ch,ks,ks)
        
        #input.backward()
        pad_grad = torch.zeros(bs, ch, x+2*pad_n, y+2*pad_n)
        #mid_grad = torch.zeros_like(self.pad_mat)
        mid_grad = torch.sum(ones_3dim*self.kernel_mat,dim=1).reshape(bs,dim2*dim3,ch,ks,ks)  #(bs,lens)
        for i in range(dim2*dim3):
            n3 = i % dim3
            n2 = int(i / dim3)
            pad_grad[:, :, n2:n2+ks, n3:n3+ks] += mid_grad[:, i]
        self.input.grad = pad_grad[:, :, pad_n:x+pad_n, pad_n:y+pad_n]
        
        #------------------------------------------------------------------------------------------------------------------------------
        #print(time.time() - time_start)
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
        
        self.weight = Parameter(torch.randn((out_features, in_features), **factory_kwargs ))#随机weight(outp, inp) 改变初始化方式
        if bias:
            self.bias = Parameter(torch.randn(out_features, **factory_kwargs))
            
            
    def forward(self, input):
        '''TODO'''
        #time_start = time.time()
        #input (bs,inp)  self.weight(inp,outp)
        #------------------------------------------------------------------------------------------------------
        
        self.input = input
        self.output = torch.mm(input, self.weight.T)+self.bias
        
        #------------------------------------------------------------------------------------------------------
        #time_end = time.time()
        #print(time_end - time_start)
        return self.output
    def backward(self, ones: Tensor):
        '''TODO'''
        #time_start = time.time()
        #input (bs,inp)  self.weight(inp,outp)  ones = self.output(bs,outp)
        #------------------------------------------------------------------------------------------------------
        
        # #bias.backweard
        self.bias.grad=torch.sum(ones,dim=0)
        
        #input.backward
        self.input.grad = torch.mm(ones, self.weight)
        
        #weight.backward
        self.weight.grad = torch.mm(self.input.T, ones ).T
        
        #------------------------------------------------------------------------------------------------------
        #time_end = time.time()
        #print(time_end - time_start)
        return self.input.grad

class CrossEntropyLoss():
    def __init__(self):
        pass
    def __call__(self, input, target):
        '''TODO'''
        #time_start = time.time()
        #------------------------------------------------------------------------------------------------------
        self.num = 1.e-6
        self.input = input
        self.target = target
        self.tars = self.input.size(0)
        self.Bs = input.size(0)
        
        #softmax  input(bs, labs) ---> output1(bs, labs)
        self.out_exp = torch.exp(input)+self.num #空间换时间  只加此处num可行
        self.exp_sum = (torch.sum(self.out_exp, dim=1)).reshape(self.Bs, 1)
        self.label_exp = torch.zeros(self.Bs, 1) #标签值(bs,1)
        for bs in range(self.Bs):
            self.label_exp[bs][0] = self.out_exp[bs][target[bs]]
        self.label_P = self.label_exp / (self.exp_sum)+self.num 
        
        #log -1 NLLloss output1(bs,labs) ---> self.output = scalar
        self.output = -sum(torch.log(self.label_P+self.num)) / self.Bs 
        
         
        #------------------------------------------------------------------------------------------------------
        #print(time.time() - time_start)
        return self.output
    def backward(self):
        '''TODO'''
        #time_start = time.time()
        #------------------------------------------------------------------------------------------------------
        
        """vec_bs = -1/self.Bs/self.label_P#(bs,1)
        self.input.grad = -self.out_exp*self.label_exp/(self.exp_sum**2) #(bs,labs)
        change = (self.label_exp-self.exp_sum)/self.label_exp #(bs, 1)
        for b_s in range(self.Bs):
            self.input.grad[b_s][self.target[b_s]] *= change[b_s][0]
        self.input.grad = self.input.grad*vec_bs"""
        
        self.input.grad = self.out_exp/(self.exp_sum+self.num)/self.Bs +self.num
        for b_s in range(self.Bs):
            self.input.grad[b_s][self.target[b_s]] -= 1/self.Bs
        #------------------------------------------------------------------------------------------------------
        #print(time.time() - time_start)
        return self.input.grad