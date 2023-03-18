# Dian_assignment_AI
21级材料学院冯雄飞AI算法代码仓库

#CNN-self分支是新的实现，主要使用tensor矩阵相关运算实现，相对main主支的自我实现函数速度快。

#为起对比作用，cnn-self.py是用自己函数实现的CNN算法，原cnn.py是调用torch.nn模块实现的CNN算法

#main分支中的函数是主要使用for循环的旧实现，速度慢，但是笔者还是把它留在main主支。

**CrossEntropyLoss**，**Linear**，**Conv2d**函数的forward与backward完成，但是存在些许for循环，代码执行速度低，尤其是Conv2d。但是相比主支的实现已经有可观的改进。

