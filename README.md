# RainTorch
一款从零手写实现的轻量级深度学习训练框架，自研反向模式自动微分引擎，采用 C++ 底层开发 + Python 上层封装，用于深入学习深度学习框架底层原理。项目命名源于个人对雨天的喜爱，从标量自动求导起步，逐步扩展到多维张量计算，支持线性回归、神经网络训练、模型推理等完整能力。
项目介绍
RainTorch 旨在复刻 PyTorch 这类主流深度学习框架的底层设计思路，从最简单的标量自动微分（参考 micrograd）开始，一步步搭建完整训练框架的核心组件，帮助深入理解动态计算图、链式法则反向传播、张量运算、网络层、优化器等核心原理。
本项目不仅可以用来完成基础的机器学习任务训练，后续还会扩展 CUDA 算子加速、模型序列化、Transformer 训练、离线部署推理等能力。
核心模块
1. 自动求导引擎（Autograd）
反向模式自动微分，前向传播时动态构建计算图
通过拓扑排序 + 链式法则实现梯度反向传播
支持四则运算、幂运算、ReLU 等常用算子的前向与反向梯度实现
多分支参数梯度自动累加，兼容复杂计算场景
2. 张量模块（Tensor）
多维张量封装，管理维度、步长、内存数据与梯度
实现广播机制、矩阵乘法、维度变换、转置、切片等基础操作
底层基于 OpenBLAS 加速矩阵运算，预留 CUDA 扩展接口
3. 模型网络模块
基类 Module：自动收集管理所有可训练参数，支持嵌套网络结构
内置常用网络层：线性层 Linear、ReLU、层归一化 LayerNorm、Dropout
可自由搭建 MLP、简易 Transformer Encoder 等网络结构
4. 损失函数
回归任务：均方误差损失 MSELoss
分类任务：交叉熵损失 CrossEntropyLoss、二分类损失 BCEWithLogitsLoss
5. 优化器
随机梯度下降 SGD、Adam 自适应优化器
支持权重衰减、梯度裁剪、学习率调度策略
6. 模型推理模块
训练权重序列化保存与加载
推理模式关闭计算图，仅执行前向计算，降低内存开销
技术栈
底层核心：C++17 实现张量、自动微分引擎
上层接口：Python 绑定，方便快速算法验证与实验
数值加速：OpenBLAS 矩阵运算加速
工程构建：CMake 跨平台编译
快速开始
1. 拉取代码
bash
运行
git clone git@github.com:nswdbbnswda/RainTorch.git
cd RainTorch
2. 编译项目
bash
运行
mkdir build && cd build
cmake ..
make -j$(nproc)
3. 运行线性回归示例
bash
运行
# Python 示例
python examples/linear_regression.py

# C++ 原生示例
./bin/linear_regression
训练示例：线性回归
python
运行
from raintorch import Tensor, Parameter, Module
from raintorch.nn import Linear
from raintorch.optim import SGD
from raintorch.loss import MSELoss

# 构建单层线性模型
model = Linear(1, 1)
# 定义优化器与损失函数
optimizer = SGD(model.parameters(), lr=1e-3)
criterion = MSELoss()

# 训练循环
for epoch in range(1000):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()  # 梯度清零
    loss.backward()        # 反向传播求梯度
    optimizer.step()       # 参数更新

    if epoch % 100 == 0:
        print(f"迭代轮次: {epoch}, 损失值: {loss.item():.4f}")
开发路线规划
 标量反向自动微分引擎
 SGD 优化器 + MSE 损失函数
 线性回归、多层感知机训练 Demo
 多维张量、广播机制、矩阵乘法实现
 Adam 优化器、LayerNorm、Dropout 层
 Transformer Encoder 实现
 CUDA GPU 算子加速
 模型权重序列化与离线部署推理
参考致谢
micrograd：标量自动求导实现灵感来源
PyTorch：框架模块化设计、自动微分架构参考
开源协议
本项目基于 MIT 协议开源。
