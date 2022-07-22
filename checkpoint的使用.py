import torch
import torch.nn as nn
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from torch import optim

# 准备数据集
# 训练集
train_set = mnist.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
# 测试集
test_set = mnist.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
# 训练集载入器
train_data = DataLoader(train_set, batch_size=64, shuffle=True)  # 通常训练集要打乱数据
# 测试集载入器
test_data = DataLoader(test_set, batch_size=128, shuffle=False)   # 测试集看情况吧，一般shuffle可能都是False

# 可视化数据
import random
for i in range(4):
    ax = plt.subplot(2, 2, i+1)
    idx = random.randint(0, len(train_set))
    digit_0 = train_set[idx][0].numpy()
    digit_0_image = digit_0.reshape(28, 28)
    ax.imshow(digit_0_image, interpolation="nearest")
    ax.set_title('label: {}'.format(train_set[idx][1]), fontsize=10, color='black')
plt.show()

class Net(nn.Module):
    def __init__(self, in_num = 784, out_num = 10):
        super().__init__()
    # 输入层的节点个数为784,FC1的节点个数为512,FC2的节点个数为256,FC3的节点个数为128,输出层的节点个数是10（分类10个数）
    # 每个全连接层后都接一个激活函数，这里激活函数选用Relu
        self.wc1 = nn.Linear(in_num, 512)
        self.ac1 = nn.ReLU(inplace = True)  # 设置为True可以节省内存，小技巧
        self.wc2 = nn.Linear(512, 256)
        self.ac2 = nn.ReLU(inplace = True)
        self.wc3 = nn.Linear(256, 128)
        self.ac3 = nn.ReLU(inplace = True)
        self.wc4 = nn.Linear(128, out_num)
    
    def forward(self, x):
        x = self.ac1(self.wc1(x))
        x = self.ac2(self.wc2(x))
        x = self.ac3(self.wc3(x))
        x = self.wc4(x)
        
        return x

net  = Net()
print(net)
# MNIST图像的维度是28 x 28 x 1=784，所以，直接将28 x 28的像素值展开平铺为 784 x 1的数据输入给输入层
# 设置优化器
# weight decay（权值衰减）最终目的是防止过拟合，其实就是正则化
optimizer = optim.SGD(net.parameters(), lr=1e-2, weight_decay=5e-4)

# 定义损失函数，分类问题显然使用交叉熵
criterion = nn.CrossEntropyLoss()


# 什么是checkpoint
## checkpoint是用来保存模型、模型权重、epoch、优化器状态、loss情况等等，比之前单纯的保存功能更多点
## 一个好处是当我们中间停止了训练之后，后续开机还能从上次的checkpoint恢复训练，很爽！！！

# 定义保存checkpoint函数
def save_checkpoint(model, epoch, loss, checkpoint_interval=1):
    if (epoch+1) % checkpoint_interval == 0:

        checkpoint = {"model_state_dict": model.state_dict(),   # 模型权重
                      "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态
                      "epoch": epoch,
                      "loss": loss}
        path_checkpoint = "./checkpoint_{}_epoch.pkl".format(epoch+1)    # 存储路径
        torch.save(checkpoint, path_checkpoint)   

# 假如我们在其中一次训练的时候出现意外中断，则可以通过checkpoint接着上次训练
'''
    path_checkpoint = "./checkpoint_4_epoch.pkl"   # 要加载的最后一次checkpoint
    checkpoint = torch.load(path_checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']   # 下次开始的epoch起点
'''
# 开始训练
# 记录训练损失
losses = []
# 记录训练精度
acces = []
# 记录测试损失
eval_losses = []
# 记录测试精度
eval_acces = []
# 设置迭代次数
nums_epoch = 10
for epoch in range(nums_epoch):
    train_loss = 0
    train_acc = 0
    net = net.train()
    for batch, (img, label) in enumerate(train_data):
        img = img.reshape(img.size(0), -1)
        img = Variable(img)
        label = Variable(label)

        # 前向传播
        out = net(img)
        loss = criterion(out, label)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录误差
        train_loss += loss.item()
        # 计算分类的准确率
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        if (batch + 1) % 200 ==0:
            print('[INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f}'.format(epoch + 1,
                                                                                 batch+1,
                                                                                 loss.item(),
                                                                                 acc))
        train_acc += acc

    save_checkpoint(net, epoch, loss)
    losses.append(train_loss / len(train_data))
    acces.append(train_acc / len(train_data))

    eval_loss = 0
    eval_acc = 0
    # 测试集不训练
    net.eval()
    for img, label in test_data:
        img = img.reshape(img.size(0),-1)
        img = Variable(img)
        label = Variable(label)

        out = net(img)
        loss = criterion(out, label)
        # 记录误差
        eval_loss += loss.item()

        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]

        eval_acc += acc
    eval_losses.append(eval_loss / len(test_data))
    eval_acces.append(eval_acc / len(test_data))

    print('[INFO] Epoch-{}: Train: Loss-{:.4f}, Accuracy-{:.4f} | Test: Loss-{:.4f}, Accuracy-{:.4f}'.format(
        epoch + 1, train_loss / len(train_data), train_acc / len(train_data), eval_loss / len(test_data),
        eval_acc / len(test_data)))