import torch
import torch.nn as nn
import torch.optim as optim

# 1. 定义神经网络（深度学习模型：多层+非线性激活）
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一层：输入1个特征 → 映射到3个隐藏神经元
        self.layer1 = nn.Linear(in_features=1, out_features=3)
        # 第二层：3个隐藏神经元 → 输出1个结果
        self.layer2 = nn.Linear(in_features=3, out_features=1)
        # 非线性激活函数，没有它就不算深度学习
        self.relu = nn.ReLU()

    def forward(self, x):
        # 前向传播（等价你手动写：h = w1*x + b1 → 激活 → y=w2*h + b2）
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# 2. 构造数据集，必须是 [样本数, 特征数] 形状
x_data = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
y_data = torch.tensor([[5.], [7.], [9.], [11.], [13.]])

# 3. 初始化网络、损失函数、优化器
model = Net()
# 均方误差损失，等价你手写的 square 求和
loss_func = nn.MSELoss()
# SGD优化器，管理网络里所有w、b参数
optimizer = optim.SGD(model.parameters(), lr=0.008)

# 4. 训练主循环
epoch_total = 2000
for epoch in range(epoch_total):
    # ① 前向传播，构建计算图
    y_pred = model(x_data)
    # ② 计算损失
    loss = loss_func(y_pred, y_data)

    # ③ 梯度清零 → 反向传播求梯度 → 更新参数
    optimizer.zero_grad()   # 对应你 zero_grad(w,b)
    loss.backward()         # 对应你手写拓扑排序+链式求导
    optimizer.step()        # 对应你 w = w - lr * w.grad

    # 打印日志
    if epoch % 100 == 0:
        print(f"Epoch:{epoch:4d} | Loss:{loss.item():.4f}")

# 5. 保存训练好的模型权重（离线部署用）
torch.save(model.state_dict(), "nn_model.pth")
print("\n模型权重已保存至 nn_model.pth")

# 6. 加载模型 + 推理预测（推理阶段不需要反向、不需要计算图）
new_model = Net()
new_model.load_state_dict(torch.load("nn_model.pth"))

# 关闭梯度计算，节省内存速度更快
with torch.no_grad():
    test_x = torch.tensor([[6.0]])
    pred = new_model(test_x)
    print(f"输入x=6，网络预测结果：{pred.item():.2f}")
    print(f"真实理论结果 y=2*6+3 = 15")
