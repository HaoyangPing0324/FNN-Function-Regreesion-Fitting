"""
@Author  : 平昊阳
@Email   : pinghaoyang0324@163.com
@Time    : 2025/12/13
@Desc    : 全连接神经网络进行回归任务，并且可视化结果
@License : MIT License (MIT)
@Version : 1.0

"""

####    导入库
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

####    是否使用GPU训练
def check_device():
    ## 检查是否有可用的 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    ## 如果是 GPU，显示详细的 GPU 信息
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))

    return device

####    数据生成
def data_generation(lower=-6, upper=6, N=200, noise_std=0.1, random_seed=42):
    ## 在区间[-6,6]内随机生成N = 200个二维输入数据样本x
    np.random.seed(random_seed)
    x = np.random.uniform(lower, upper, size=(N, 2)).astype(np.float32)

    ## 根据给定函数计算y
    x1, x2 = x[:, 0], x[:, 1]
    y1 = np.cos(x1) + np.sin(x2)                                                # y1 = cos(x1) + sin(x2)
    y2 = np.log10(1 + x1 ** 2 + x2 ** 2)                                        # y2 = log(1 + x1^2 + x2^2)
    y3 = np.sqrt(1 + x1 ** 2 + x2 ** 2)                                         # y3 = sqrt(1 + x1^2 + x2^2)
    y = np.stack([y1, y2, y3], axis=1).astype(np.float32)                 # 合并目标值(N, 3)

    ## 使用np.random.normal(0, std=***, size=***)给y随机添加噪声，以模拟真实数据
    noise = np.random.normal(0, noise_std, size=y.shape).astype(np.float32)
    y += noise                                                                   # 向 y 添加噪声

    return x,y

def dataset_generation(x , y , test_size = 0.3, random_seed=42):
    ##  将最终的数据制作成pytorch数据集，其中70%用于训练，30%用于验证。
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state=random_seed)
    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_data = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    return train_data , val_data

####    模型设计
class Net(torch.nn.Module):
    def __init__(self, n_input = 2 , n_hidden = [64,64], n_output = 3 , AF='relu'):
        super(Net, self).__init__()

        ## 建一个全连接神经网络，包含至少两层全连接层（层数L可自行实验测试，每一层的神经元数也自行尝试选择）
        layers = []
        if not n_hidden:
            n_hidden = [64, 64]
            raise ValueError("n_hidden为空")

        ## 输入层输入2个特征
        in_dim = n_input

        ## 构建隐藏层（+激活函数）：隐藏层中激活函数使用ReLU或Sigmoid（可自行测试）
        for h in n_hidden:
            layers.append(nn.Linear(in_dim, h))
            if AF == 'relu':
                layers.append(nn.ReLU())                        # ReLU 激活函数
            elif AF == 'sigmoid':
                layers.append(nn.Sigmoid())                     # Sigmoid 激活函数
            else:
                raise ValueError("激活函数选择无效！请选择 'relu' 或 'sigmoid'.")
            in_dim = h

        ## 构建输出层:输出层输出3个特征
        layers.append(nn.Linear(n_hidden[-1], n_output))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

####    训练过程
def train(model, train_data, val_data, device, optimizer_choice='Adam', epochs=100, batch_size=32, lr=0.001):
    ## 数据加载:置合适的batch size
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    ## 损失函数:选用均方误差（MSE）
    loss_func = torch.nn.MSELoss()                                  # 预测值和真实值的误差计算公式 (均方差)

    ## 选择优化器:优化器可以选用 Adam 或 SGD，设置合适的学习率
    if optimizer_choice == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)           # Adam 优化器
    elif optimizer_choice == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr)            # SGD 优化器
    else:
        raise ValueError("优化器选择无效！请选择 'Adam' 或 'SGD'.")

    ## 训练若干个epoch，直至损失收敛，记录每个epoch的训练损失和验证损失
    train_losses, val_losses = [], []                               # 存储每个 epoch 的训练损失和验证损失
    for epoch in range(epochs):
        ## 训练阶段
        model.train()                                               # 设置模型为训练模式
        running_train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()                                   # 清空之前的梯度
            outputs = model(xb)                                     # 前向传播
            loss = loss_func(outputs, yb)                           # 计算损失
            loss.backward()                                         # 反向传播
            optimizer.step()                                        # 更新模型参数
            running_train_loss += loss.item() * xb.size(0)          # 累计训练损失
        train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        ## 验证阶段
        model.eval()                                                # 设置模型为评估模式
        running_val_loss = 0.0
        with torch.no_grad():                                       # 在验证阶段不计算梯度
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)
                loss = loss_func(outputs, yb)
                running_val_loss += loss.item() * xb.size(0)
        val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        ## 输出当前 epoch 的训练损失和验证损失
        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}")

    return train_losses, val_losses

####    结果展示
### 利用matplotlib 绘制训练过程中损失（训练损失和验证损失）随 epoch 变化的曲线
def plot_loss_curve(train_losses, val_losses, epochs):
    # 绘制线性坐标图
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train and Validation Loss over Epochs (Linear Scale)")
    plt.grid(True)
    plt.show()

    # 绘制对数坐标图
    plt.figure(figsize=(8, 6))
    plt.plot(range(epochs), train_losses, label="Train Loss")
    plt.plot(range(epochs), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Train and Validation Loss over Epochs (Log Scale)")
    plt.yscale('log')  # 设置 y 轴为对数坐标
    plt.grid(True)
    plt.show()

### 再次生成少量数据（例如20），比较展示模型预测值与真实（带噪声）目标值的差异（可采用散点图或折线图）
def plot_pred_compare(model, device, noise_std=0.1):
    model.eval()  # 设置模型为评估模式

    # 生成 20 个新的数据样本，并计算真实目标值（带噪声）
    x_new, y_true = data_generation(N=20,noise_std = noise_std)
    # 将新数据移动到设备上
    x_new = torch.tensor(x_new, dtype=torch.float32).to(device)

    # 使用模型进行预测
    with torch.no_grad():
        y_pred = model(x_new).cpu().numpy()

    # 绘制真实值和预测值的差异
    plt.figure(figsize=(12, 4))
    for i in range(3):  # 3个输出维度：y1, y2, y3
        plt.subplot(1, 3, i + 1)
        plt.scatter(range(20), y_true[:, i], label='True', color='tab:blue', alpha=0.6)
        plt.scatter(range(20), y_pred[:, i], label='Predicted', color='tab:red', marker='x')
        plt.title(f'Output dim {i + 1}')
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.show()

####    主函数
def main():
    ##  定义参数
    noise_std = 0.001                                       # 噪声标准差
    random_seed = 42                                        # 随机种子
    n_hidden = [64, 64, 64]                                 # 隐藏层神经元数
    AF = 'relu'                                             # 激活函数
    epochs = 300                                            # 训练轮数
    batch_size = 32                                         # 每个批次的样本数量
    lr = 0.001                                              # 学习率
    optimizer_choice = 'Adam'                               # 优化器可以选用 Adam 或 SGD
    device = check_device()                                 # 是否使用 GPU 训练

    ##  回归拟合
    x,y = data_generation(noise_std=noise_std, random_seed=random_seed)                                 # 生成原始数据
    train_data,val_data = dataset_generation(x , y ,random_seed=random_seed)                            # 生成数据集
    model = Net(n_hidden= n_hidden , AF= AF ).to(device)                                                # 模型设计
    train_losses, val_losses = train(model=model, train_data=train_data, val_data=val_data, device=device,
                                     optimizer_choice=optimizer_choice,epochs=epochs,
                                     batch_size=batch_size, lr=lr)                                      # 训练过程
    plot_loss_curve(train_losses=train_losses, val_losses=val_losses, epochs=epochs)                    # 绘制损失曲线
    plot_pred_compare(model=model, device=device, noise_std=noise_std)                                  # 比较展示差异

####    运行
if __name__ == "__main__":
    main()