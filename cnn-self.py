import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision.transforms as transforms

#导入自写实现的代码
import nn.function as my
#CNN框架
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1,padding=2), #卷积层（输入通道数，卷积核个数，卷积核尺寸，卷积核步长,外延填充数）
            nn.BatchNorm2d(16),       #对这16个结果进行规范处理,卷积网络中(防止梯度消失或爆炸)，设置的参数就是卷积的输出通道数
            nn.ReLU(),       #激活函数
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = my.Linear(7*7*32, 10)
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x = x.reshape(x.size(0), -1)  
        x=self.layer3(x)
        return x

#超参数num_epochs = 5 BATCH_SIZE = 100 learning_rate = 0.001 moment = 0
num_epochs = 1
BATCH_SIZE = 5
learning_rate = 0.001
moment = 0

# 读取图像数据集
train_data = torchvision.datasets.MNIST(root ='./mnist',train = True,transform=transforms.ToTensor(),download = True )
train_loader=Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# 读取标签数据集   

test_data=torchvision.datasets.MNIST(root='./mnist', train=False,transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)
"""with torch.no_grad(): #取消计算图，避免内存消耗，但是不方便模块化管理
    images_test=Variable(torch.unsqueeze(test_data.data, dim=1)).type(torch.FloatTensor)[:]/255   
    labels_test=test_data.targets[:]"""



#实例化CNN模型，并且将模型导入到cuda核心中
model = CNN()


#损失函数 交叉熵 
Lossfunc = my.CrossEntropyLoss()  
#优化函数有Adam和SGD常用，这里选择SGD
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

#开始训练模型
for epoch in range(num_epochs):
    #加载训练数据
    for step,(x,y) in enumerate(train_loader):
        #分别得到训练数据的x和y的取值
        ima_train=Variable(x)
        lab_train=Variable(y)
        
        optimizer.zero_grad() 
        output=model(ima_train)         #调用模型输出结果
        loss=Lossfunc(output,lab_train) #计算损失值
        loss.backward()         #反向传播
        optimizer.step()          #梯度下降
 
        #每执行100次，输出一下当前epoch、loss、accuracy
        if(step+1) % 5 == 0:
            print('Epoch [%d/%d], Iter[%d/%d] Loss: %.4f' %(epoch, num_epochs, step+1, len(train_data)/BATCH_SIZE, loss.item()))

model.eval()  #改为预测模式
correct ,total= 0,0
for images, labels in test_loader:
    images = Variable(images)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)  #按照维度取最大值，返回每一行中最大的元素，且返回索引
    total += labels.size(0)         #labels.size(0) = 100 = batch_size
    correct += (predicted.cpu() == labels).sum()  #计算每次批量处理后，100个测试图像中有多少个预测正确，求和加入correct
       
print('Test accuracy of the model on the 10000 test images: %d %%' %(100 * correct/total))