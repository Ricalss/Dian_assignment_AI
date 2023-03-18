# Dian_assignment_AI

21级材料学院冯雄飞AI算法代码仓库

#分支CNN-self是最新的实现，主要使用tensor矩阵相关运算实现，代码执行效率更高，请移步至分支CNN-self

#main分支中的函数实现主要使用for循环，速度慢，但是笔者还是把它留在main主支,留作记录

#CNN框架已经实现，可见cnn.py文件

**CrossEntropyLoss**，**Linear**，**Conv2d**函数的forward与backward完成，但是for循环过多，代码执行速度十分低，尤其是Conv2d。

my.CrossEntropyLoss 与my.Linear能正常加入cnn-self.py中执行，速度慢。

my.Conv2d加入到cnn-self.py中，会出现闪退，没有错误提示，但是在module_test.py中能够通过较小的随机张量的测试，但是速度很慢。笔者未继续改良，在分支CNN-self中有新的实现。

后续改良算法，可以考虑用多用矩阵和向量计算。