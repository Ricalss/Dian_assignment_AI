import random
import argparse
import functools

import numpy as np
import torch
import torch.nn.functional as F

from nn.function import *
import nn

#命令行工具
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('tests', nargs='*')
    return parser.parse_args()

#返回n个N分布的整数，介于a-b
def randnint(n, a=8, b=16):
    """Return N random integers."""
    return (random.randint(a, b) for _ in range(n))

#"partial" 在 "functools" 模块中，可以用来固定一个函数的部分参数，并返回一个新的函数，该函数可以被调用并使用剩余的参数。
#isclose判断数据是否相近rtol相对误差，atol绝对误差
isclose = functools.partial(np.isclose, rtol=1.e-5, atol=1.e-5)

#是为后续三个test类的父类，提供一些基本测试功能
class TestBase(object):
    def __init__(self, module, input_shape, module_params=None):
        self.module = module.split('.')[-1]
        module_params = module_params.split(',') \
                        if module_params is not None else []
        input_shape = input_shape.split('x')
        keys = set(module_params + input_shape)#集合
        args = {k: v for k, v in zip(keys, randnint(len(keys)))}#随机维度， zip打包成列表，每个元素是元组，元组包含传递的可迭代参数
        args['k_s']=3
        #args['L'],args['B']=8 ,13
        #args = {'k_s': 3, 'B': 8, 'Cp': 16, 'H': 15, 'C': 9, 'W': 13}#给定维度原：{"W":4,"Cp":5,"B":2,"H":4,"C":3,"k_s":2,"L":5},L为线性层输入
        print(module,args)
        self.nnt = 0.9*torch.rand(tuple(args[k] for k in input_shape))+0.1#放缩产生随机数的范围，
        self.ptt = self.nnt.clone().detach()#detach从当前计算图中分离下来的；深拷贝但是无梯度
        self.ptt.requires_grad = True 
        self.nnt.requires_grad = True 
        self.nnm = getattr(nn.function,module)(*tuple(args[k] for k in module_params))
    def forward_test(self):
        # self.pt_out = ... 在子类定义的同名函数中，输出是为pytorch输出
        self.nn_out = self.nnm(self.nnt) #得到nn/function中定义的输出
        if self.nn_out is None:
            print('your forward output is empty')
            return False
        res = isclose(self.nn_out.cpu().detach().numpy(),
                       self.pt_out.cpu().detach().numpy()).all().item()#比较
        if res :
            return True
        else:
            print('forward_dismatch:') 
            print("pytorch result:",self.pt_out.cpu().detach().numpy())
            print("your result:",self.nn_out.cpu().detach().numpy())
            return False
        
    def backward_test(self):
        self.pt_out.sum().backward()#pytorch求loss需要收敛到一个值所以这里做一个求和
        self.pt_grad = self.ptt.grad 
        if self.nn_out is None:
            print("your backward output is empty")
            return False
        self.nn_grad = self.nnm.backward(torch.ones_like(self.nn_out))#传回一个全一矩阵，相当于forward的输出label
        if self.nn_grad is None:
            print("your backward grad is empty")
            return False
        res = isclose(self.nn_grad.detach().numpy(), self.pt_grad.detach().numpy()).all().item() #原文件self.nn_grad后添加.detach().numpy()
        if res :
            return True
        else:
            print('backward_dismatch:') 
            print("pytorch result:",self.pt_grad.detach().numpy())
            print("your result:",self.nn_grad)
            return False


class Conv2dTest(TestBase):
    def __init__(self):
        super().__init__('Conv2d', input_shape='BxCxHxW', module_params='C,Cp,k_s')#调用父类初始化函数初始化类

    def forward_test(self):

        self.pt_wgt = torch.Tensor(self.nnm.weight.data) #提取自定义的卷积层随机初始化的数据
        self.pt_wgt.requires_grad = True
        self.pt_bias = torch.Tensor(self.nnm.bias.data)
        self.pt_bias.requires_grad = True
        #得到pytorch计算结果
        self.pt_out = F.conv2d(input=self.ptt, weight=self.pt_wgt,
                               bias=self.pt_bias,stride=1,padding=0) #padding=0
        return super().forward_test()

    def backward_test(self):
        s = super().backward_test()
        s &= isclose(self.nnm.weight.grad.detach().numpy(), self.pt_wgt.grad   #原文件self.nn后添加.detach().numpy()
                     .detach().numpy()).all().item()
        s &= isclose(self.nnm.bias.grad.detach().numpy(), self.pt_bias.grad    #原文件self.nnm后添加.detach().numpy()
                     .detach().numpy()).all().item()
        return s
    

class LinearTest(TestBase):
    def __init__(self):
        super().__init__('Linear', input_shape='BxL', module_params='L,C')

    def forward_test(self):
        self.pt_wgt = self.nnm.weight.clone().detach()
        self.pt_wgt.requires_grad = True
        self.pt_bias = self.nnm.bias.clone().detach()
        self.pt_bias.requires_grad = True
        self.pt_out = F.linear(input=self.ptt, weight=self.pt_wgt,
                               bias=self.pt_bias)
        return super().forward_test()

    def backward_test(self):
        s = super().backward_test()
        if s:
            s = isclose(self.nnm.weight.grad.detach().numpy(), self.pt_wgt.grad
                     .detach().numpy()).all().item()
            if s == False:
                print('weight_grad_dismatch')
        if s:
            s = isclose(self.nnm.bias.grad.detach().numpy(), self.pt_bias.grad
                     .detach().numpy()).all().item()
            if s == False:
                print('bias_grad_dismatch')
        return s

class CrossEntropyTest(TestBase):
    def __init__(self):
        super().__init__('CrossEntropyLoss', input_shape='BxL')

    def forward_test(self):
        pt_crossentropyloss = torch.nn.CrossEntropyLoss()
        batch = self.nnt.size(0)
        self.test_target = torch.ones(batch,dtype=torch.int64)#选取的label为1
        self.pt_out = pt_crossentropyloss(input=self.ptt, target=self.test_target)
        self.nt_out = self.nnm(input=self.nnt, target=self.test_target)
    
        res = isclose(self.pt_out.detach().numpy(),self.nt_out.detach().numpy()).all().item()
        if res:
            return True
        else:
            print('crossentropy_forward_dismatch:') 
            print("pytorch result:",self.pt_out.detach().numpy())
            print("your result:",self.nt_out.detach().numpy())
            return False
    def backward_test(self):
        self.pt_out.backward()
        self.nn_grad = self.nnm.backward()
        res = isclose(self.ptt.grad.detach().numpy(),self.nn_grad.detach().numpy()).all().item()
        if res:
            return True
        else:
            print('crossentropy_backward_dismatch:') 
            print("pytorch result:",self.ptt.grad.detach().numpy())
            print("your result:",self.nn_grad.detach().numpy())
            return False

if __name__ == "__main__":
    test_list = [LinearTest()]#,CrossEntropyTest(),Conv2dTest()
    for a in test_list:
        print("Test",a.module)
        print("forward:",a.forward_test())
        #print("backward:",a.backward_test())
