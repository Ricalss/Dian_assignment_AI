
import torch
x = torch.ones(3,4,5)
print(sum(x))
print(x.sum())
print(torch.sum(x))
a = torch.ones(3,4)
b = torch.ones(5,4,2)
print(torch.matmul(a, b))
print(torch.matmul(a, b).shape)
bs ,ch, kn, ks=6, 3, 16, 5
dim2, dim3 =10 ,10
tensor1 = torch.randn(bs, dim2,dim3,ch*ks*ks)  
tensor2 = torch.randn(kn, ch*ks*ks)
print(torch.matmul(tensor1, tensor2.T).size())
tensor3 = torch.randn(bs, kn, dim2, dim3)

x = torch.tensor([[1,2], [4,5], [7,8]])
print(x.expand(3,6))