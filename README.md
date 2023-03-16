# Dian_assignment_AI
21级材料学院冯雄飞AI算法代码仓库

CNN框架已经实现，可见cnn.py文件
**CrossEntropyLoss**，**Linear**，**Conv2d**函数的forward与backward完成，但是for循环过多，代码执行速度十分低，尤其是Conv2d。

my.CrossEntropyLoss 与my.Linear能正常加入cnn-self.py中执行，速度慢。

my.Conv2d加入到cnn-self.py中，会出现闪退，没有错误提示，但是在module_test.py中能够通过较小的随机张量的测试，但是速度很慢。

后续改良算法，可以考虑用多用矩阵和向量计算。