### level 1
#配置Python环境，如果个人笔记本有高性能GPU（即显卡），可以安装cuda使用GPU加速训练。安装torch、pandas、numpy等常用数据处理、机器学习的相关库。
#在cnn.py文件中使用**torch**库完成**mnist数据集**手写数字识别（至少包含卷积层、线性层，损失函数使用交叉熵，准确率达到90%以上）
import pandas
import numpy as np
import torch
import tqdm

def load_mnist(root ='./Dian2023/MNIST_data/', n_samples=int(6e4)):

    # 读取标签数据集        
    with open(root + 'train-labels.idx1-ubyte', 'rb') as labelpath:
        labels_train = torch.tensor(np.fromfile(labelpath, dtype=np.uint8))
    with open(root + 't10k-labels.idx1-ubyte', 'rb') as labelpath:
        labels_test = torch.tensor(np.fromfile(labelpath, dtype=np.uint8))
    # 读取图像数据集
    rows = 28
    cols = 28
    with open(root + 'train-images.idx3-ubyte', 'rb') as imagepath:
        images_train = torch.tensor(np.fromfile(imagepath, dtype=np.uint8).reshape(60000,rows*cols),dtype = torch.float)
    with open(root + 'train-images.idx3-ubyte', 'rb') as imagepath:
        images_test = torch.tensor(np.fromfile(imagepath, dtype=np.uint8).reshape(10000,rows*cols),dtype = torch.float)
    return (images_train, labels_train, images_test, labels_test, rows*cols)
#损失函数 交叉熵 下使用torch.CrossEntropyLoss()
criterion = torch.nn.CrossEntropyLoss()  

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)  torch自带优化器,本次不使用，采用小样本训练

images_train, labels_train, images_test, labels_test, scale = load_mnist()
#初始化weight
weight = torch.randn(10,scale)

#优化步进
step_size = 0.01

#小样本训练
times =0
while tqdm(times==100):
    data_batch = torch.sample_training_data(images_train, 256) #  256个样本
    weights_grad = torch.evaluate_gradient(criterion, data_batch, weight)
    weight += - step_size * weights_grad
    times += 1

#正确率评判
correct = 0
for i in range(len(images_test)):
    if labels_test[i] == torch.argmax(torch.mm(weight,images_test[i])):
        correct += 1
print('==> correct:', correct)
print('==> total:', len(images_test))
print('==> acc:', correct / len(images_test))