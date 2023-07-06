# imageRecognition

本仓库分为两个项目

## 虚假人脸识别

### 原理

  ResNet-18是一种经典的CNN网络，是 Deep Residual Learning 团队在 2017 年提出的。它是为了解决 ImageNet 数据集上的图像分类任务而设计的，是目前最先进的图像分类模型之一。  ResNet-18 具有 18 个卷积层和 6 个全连接层，与传统的卷积神经网络相比，它在深度和广度上都有更高的分辨率和更好的性能。具体来说，它具有更大的池化层、更多的嵌入层和更密集的全连接层，可以捕捉更多的特征信息。  残差网络的核心思想是：每个附加层都应该更容易地包含原始函数作为其元素之一  ResNet沿用了VGG完整的3×3卷积层设计。 残差块里首先有2个有相同输出通道数的3×3卷积层。 每个卷积层后接一个批量规范化层和ReLU激活函数。 然后我们通过跨层数据通路，跳过这2个卷积运算，将输入直接加在最后的ReLU激活函数前。 这样的设计要求2个卷积层的输出与输入形状一样，从而使它们可以相加。 如果想改变通道数，就需要引入一个额外的1×1卷积层来将输入变换成需要的形状后再做相加运算。  

![image-20230705012633733](./assets/image-20230705012633733.png)

  在它的训练过程采用了一种称为“无损编码”的技术，可以减少传输数据的量，从而提高效率和性能。相对于之前的图像分类网络，ResNet-18 在性能、分辨率、参数数量和泛化性能方面表现出色，特别适合处理大规模图像数据集和复杂的分类任务。  

因此我们将实验数据集按照实验要求进行分类和划分后，训练一个ResNet-18网络 ，再用验证集进行测试。

### 步骤

#### **数据集划分**  

对数据进行清洗和预处理后，将其中 500 张真实人脸和 500 张虚假人脸作为训练集，其余作为测试集  

```python
from __future__ import print_function, division
import os
import random,shutil
# 给定一个人脸数据集，其中包含1999张真实人脸，1999张虚假人脸。
# 将其中500张真实人脸和500张虚假人脸作为训练集，其余作为测试集

#数据路径
# real_face_path='./CNN_synth_testset/0_real'
# fake_face_path='./CNN_synth_testset/1_fake'
face_dataset_path='./CNN_synth_testset'

#获取文件夹下所有文件名
def listDir(path):
    file_list=os.listdir(path)
    return file_list
#创建文件夹
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

#划分训练集和测试集和验证集
divided_data_path='./face_data_divided/'
def datasetDivide(divide_path):
    # 读取真实人脸和虚假人脸数据路径
    path=os.listdir(face_dataset_path)
    for doc in path:
        data=os.listdir(os.path.join(face_dataset_path,doc))
        #将人脸数据清洗
        random.shuffle(data)
        train_face_data=data[:500]
        valid_face_data=data[500:1999]
        # test_face_data=[]#
        mkdir(divide_path+'train/'+doc+'/')
        mkdir(divide_path+'validation/'+doc+'/')
        # mkdir(divide_path+'test/'+doc+'/')
        for train_face in train_face_data:
            shutil.copy(os.path.join(face_dataset_path,doc,train_face),
                        divide_path+'train/'+doc+'/'+train_face)
            
        for valid_face in valid_face_data:
            shutil.copy(os.path.join(face_dataset_path,doc,valid_face),
                        divide_path+'validation/'+doc+'/'+valid_face)
            
    print('数据集划分完成')
    
datasetDivide(divided_data_path)

```

![image-20230705012813749](./assets/image-20230705012813749.png)

#### **数据预处理和超参数设置**

```python
超参数设置
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import torch.backends.cudnn as cudnn
import numpy as np
import os

# 超参数
learning_rate = 0.001
batch_size = 4
num_epochs = 20

cudnn.benchmark = True#加速卷积神经网络的训练

数据集预处理
# 数据预处理——数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(224),  # 随机裁剪
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),  # 缩放为256×256
        transforms.CenterCrop(224),  # 中心裁剪为224×224
        transforms.ToTensor(),  # 转换为张量，且值缩放到[0,1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ]),
}

#加载数据集
data_path=divided_data_path
image_datasets = {x: datasets.ImageFolder(os.path.join(divided_data_path, x),
                                            data_transforms[x])
                    for x in ['train', 'validation']}
# print(image_datasets)
#数据的批量加载
#num_workers=0表示单线程，因为在windows不支持多线程
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,shuffle=True, num_workers=0)
                for x in ['train', 'validation']}
#数据集大小
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
# print(dataset_sizes)
#类别名称
class_names = image_datasets['train'].classes
# print(class_names)
#GPU是否可用
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

```

####   **数据集可视化**

```python
import matplotlib.pyplot as plt
import numpy as np

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

def plot_images(images, labels, preds=None):
    """
    Plot a batch of images with their labels and predictions (if given).

    Args:
        images (tensor): A batch of images.
        labels (tensor): The true labels for each image.
        preds (tensor, optional): The predicted labels for each image. Defaults to None.
    """
    # Set up the figures
    num_images = len(images)
    cnt=0
    plt.figure(figsize=(16, 16))
    # Plot each image
    for i in range(num_images):
        cnt+=1
        plt.subplot(1, num_images, cnt)
        plt.xticks([], [])#设置x轴刻度
        plt.yticks([], [])#设置y轴刻度
        # Set the title
        if preds is not None:
            title = f"predicted: {class_names[preds[i]]}, true: {class_names[labels[i]]}\n"
            color = 'green' if preds[i] == labels[i] else 'red'
        else:
            title = f"{class_names[labels[i]]}\n"
            color = 'green'        
        plt.title(title, color=color)
        # Unnormalize the image
        image = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        plt.imshow(image)
        

    plt.show()

plot_images(inputs, classes)

```

![image-20230705012939724](./assets/image-20230705012939724.png)

####   **模型构建及训练**

  我们定义了一个train_model函数，用于训练神经网络模型。函数接收模型（model）、损失函数（criterion）、优化器（optimizer）、学习率调整器（scheduler）和训练的轮数（num_epochs）作为参数，在该函数内部，进行模型的训练和验证，并记录训练和验证损失（loss）和准确率（accuracy）。其中我们使用tqdm库来包装我们的训练参数epoch,来显示进度条。  在训练过程我们使用了预训练模型、交叉熵损失函数、随机梯度下降法优化器以及学习率调整器，通过训练和验证来调整模型参数，最终得到一个表现较好的模型。  

```python
from tqdm import tqdm
from torch.optim import lr_scheduler
import copy,time
def train_model(model, criterion, optimizer,scheduler,num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    start_time = time.time()
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    pbar=tqdm(range(num_epochs))
    for epoch in pbar:
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*10}')
        
        # 每个 epoch 都有一个培训和验证阶段
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # 将模型设置为训练模式
            else:
                model.eval()   # 将模型设置为评估模式
            
            running_loss = 0.0#记录损失
            running_corrects = 0#记录正确的个数
            
            # 包装器tqdm，用于显示进度条
            # data_loader = tqdm(dataloaders[phase], desc=f'{phase} {epoch+1}/{num_epochs}')
            
            # 迭代数据
            for inputs, labels in  dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 清零参数梯度
                optimizer.zero_grad()

                # 前向传递
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 只有在训练阶段才进行反向传递+优化
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # 更新学习率
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                scheduler.step()#对优化器的学习率进行调整
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)
                
            # print(f'{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}')
            
            pbar.set_postfix_str(f'{phase} loss: {epoch_loss:.4f}, acc: {epoch_acc:.4f}')
            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
    
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    return model, train_loss_history, train_acc_history, val_loss_history, val_acc_history

#加载预训练模型
model_ft = models.resnet18(pretrained=True)

#替换最后一层的分类器
#数据集只有两个类别（真实人脸和虚假人脸），因此新的全连接层的输出维度为2。
num_features=model_ft.fc.in_features
model_ft.fc=nn.Linear(num_features,len(class_names))

#定义损失函数和优化器
model_ft=model_ft.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=optim.SGD(model_ft.parameters(),lr=learning_rate,momentum=0.9)

#学习率调整器
exp_lr_scheduler=lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

#训练模型
model_ft, train_loss_history, train_acc_history, val_loss_history, val_acc_history = train_model(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=15)

```



#### **训练过程及结果可视化**

```python
#可视化训练过程
def plot_history(train_loss_history, train_acc_history, val_loss_history, val_acc_history):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.plot(train_loss_history, label='train')
    plt.plot(val_loss_history, label='validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # plt.figure(figsize=(15, 5))
    # plt.subplot(122)
    # plt.plot(train_acc_history, label='train')
    # plt.plot(val_acc_history, label='validation')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()
    
    

def visualize_model(model, num_images=8):
    was_training = model.training
    model.eval()
    images_so_far = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['validation']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            plot_images(inputs.cpu(), preds, labels)
            images_so_far += batch_size

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
        model.train(mode=was_training)
        
visualize_model(model_ft)
```

plot_history(train_loss_history, train_acc_history, val_loss_history, val_acc_history)
主要涉及两个函数，一个是用于可视化训练过程的 plot_history() 函数，另一个是用于可视化模型的 visualize_model() 函数。
plot_history() 函数接受四个参数，分别是训练损失、训练精度、验证损失和验证精度的历史记录。该函数会将训练和验证损失随着训练轮数的增加而变化的曲线绘制在同一张图上进行对比，以便于分析模型的训练效果。最后通过 plt.show() 函数将图像展示出来。
visualize_model() 函数接受两个参数，一个是模型对象，另一个是要可视化的图片数目，默认为 8。该函数会将模型在验证集上的预测结果可视化出来。具体实现是通过一个循环遍历验证集数据集中的图片，然后将这些图片、模型的预测结果以及真实标签传递给 plot_images() 函数进行展示。在展示完指定数量的图片之后，该函数会将模型的模式切换回之前的模式，即训练模式或评估模式。
通过调用 visualize_model() 函数和 plot_history() 函数，可视化了模型的预测结果和训练过程。

**训练过程**

  首先分析在6个Epoch内，我们看到Loss函数收敛到接近于零，无论是训练的loss还是验证集的loss,验证集上的正确率达到了100%，时间花费在14分钟左右，在GPU上进行训练  

![image-20230705013142191](./assets/image-20230705013142191.png)

![image-20230705013115855](./assets/image-20230705013115855.png)

![image-20230705013209487](./assets/image-20230705013209487.png)

![image-20230705013215212](./assets/image-20230705013215212.png)





## 手写四则运算识别

> 训练数据是在MNIST 数据集上，扩充了“ “＋ 、-、×、÷、（、）” 6种符号数据集，我们称之为“MNIST +”数据集。每种样本以png图片格式保存在一个文件夹中，总共16个种类，每种含5000张以上图像样本。
>
> 总共100张测试样本图片，存储在test文件夹下。

下面我用LSTM来识别经过扩充后的手写数据集

**1)** 读取图片数据集；

我们通过listdir列举出数据集下路径下的文件夹，文件夹下有相对应的类别名和它下面的文件，我们可以把这个作为图片的标签，从而创建数据集。

```python
import os
import cv2
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.nn.utils.rnn import pack_padded_sequence
from PIL import Image
from torch.utils.data import Dataset

#读取minist plus数据集
images = []#存放图片数据
targets = []#存放图片标签

dataset_path = './mnist+/'
labels = os.listdir(dataset_path)#读取文件夹下的文件名

for label in labels:
    folder_path = os.path.join(dataset_path, label)#文件夹路径
    for img_name in os.listdir(folder_path):#读取文件夹下的文件名
        img_path = os.path.join(folder_path, img_name).replace('\\','/')#图片路径
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)#读取灰度图
        images.append(img)
        targets.append(label)
        
#输出数据集大小
print('images size: ', len(images))
print('targets size: ', len(targets))
```

由于在训练过程发现模型的泛化能力不是很高，因此做了一些数据集的扩充，主要是做翻转操作

```python
#数据增强
#手段一:缩放
def scale_image(image,scale_factor):
    width=int(image.shape[1]*scale_factor)
    height=int(image.shape[0]*scale_factor)
    scale_image=cv2.resize(image,(width,height))
    return scale_image

#手段二:平移
def translate_image(image,shift_x,shift_y):
    tarnslate_matrix=np.float32([[1,0,shift_x],[0,1,shift_y]])
    shifted_image=cv2.warpAffine(image,tarnslate_matrix,(image.shape[1],image.shape[0]))
    return shifted_image

import random
def expand_dataset(images, targets):
    #扩充数据集
    new_images = []
    new_targets = []
    for i in range(len(images)):
        img = images[i]
        target = targets[i]
        for angle in [0, 90, 180, 270]:
            new_img = np.rot90(img, angle)
            new_images.append(new_img)
            new_targets.append(target)
    return np.copy(new_images, order='C'), np.copy(new_targets, order='C')

images, targets = expand_dataset(images, targets)
# 输出数据集大小
print('images size: ', len(images))
print('targets size: ', len(targets))
```

然后为了方便数据集的处理，我定义了一个自定义数据集，并用transform进行处理

```python
#数据处理
data_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])
#自定义数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, index):
        img = self.images[index]
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.images)

dataset=MyDataset(images, targets, transform=data_transform)
images, targets = dataset.images, dataset.targets
```

接下来我们需要用train_test_split来划分训练集或者测试集(也可以把它当作验证集),batchsize这里设置的是40

```python
#划分训练集和(测试集或者验证集)
X_train, X_test, y_train, y_test = train_test_split(images, targets, test_size=0.2, random_state=42)
train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=40, shuffle=True)#训练集
test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=40, shuffle=True)#测试集
```



![img](./assets/clip_image008.jpg)

**2)** 创建深度神经网络模型；

先定义LSTM模型的结构

```python
#创建LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()#继承父类
        self.hidden_size = hidden_size#隐藏层大小
        self.num_layers = num_layers#隐藏层数
        # self.dropout = nn.Dropout(0.5)#dropout层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)#LSTM层
        self.fc = nn.Linear(hidden_size, num_classes)#全连接层
        
    def forward(self, x):
        #初始化隐藏层和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        #前向传播LSTM层
        out, _ = self.lstm(x, (h0, c0))#out的形状为(batch_size, seq_length, hidden_size)
        
        #解码最后一个时刻的隐状态
        out = self.fc(out[:, -1, :])
        return out
```

设置模型超参数，和模型的具体参数

```python
#超参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#如果有GPU则使用GPU
# model=LSTM(28, 128, 4, len(labels)).to(device)#创建模型
# criterion = nn.CrossEntropyLoss()#损失函数
# optimizer = optim.Adam(model.parameters(), lr=0.001)#优化器
model=LSTM(28, 128, 4, len(labels)).to(device)#创建模型
criterion = nn.CrossEntropyLoss()#损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)#优化器
num_epochs = 14#迭代次数
```

**3)** 训练模型；

```python
#超参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#如果有GPU则使用GPU
# model=LSTM(28, 128, 4, len(labels)).to(device)#创建模型
# criterion = nn.CrossEntropyLoss()#损失函数
# optimizer = optim.Adam(model.parameters(), lr=0.001)#优化器
model=LSTM(28, 128, 4, len(labels)).to(device)#创建模型
criterion = nn.CrossEntropyLoss()#损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)#优化器
num_epochs = 14#迭代次数

#训练模型
train_acc=[]#训练集准确率
train_loss=[]#训练集损失

for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for i, (batch_images, batch_labels) in enumerate(train_loader):
        batch_images = batch_images.reshape(-1, 28, 28).to(device).float()#将图片转换为(batch_size, seq_length, input_size)
        batch_labels = torch.tensor([labels.index(label) for label in batch_labels]).to(device)#将标签转换为数字
        # 前向传播
        outputs = model(batch_images)
        # 计算损失
        loss = criterion(outputs, batch_labels)
        # 将梯度置零，反向传播，更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 计算训练的损失和准确率
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        running_acc += torch.sum(preds == batch_labels.data)
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)
    train_acc.append(epoch_acc)
    train_loss.append(epoch_loss)
    print('Epoch [{}/{}], Loss: {:.4f}, Acc: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc))
```

模型训练输出过程

![img](./assets/clip_image016.jpg)

可以绘制我们每个轮次的损失

```python
#绘制训练集准确率曲线
import matplotlib.pyplot as plt
#将tensor转换为numpy(因为tensor不能直接画图)
train_loss=torch.Tensor(train_loss).cpu().numpy()
plt.plot(train_loss, label='train_loss')
plt.legend()
plt.show()
```

![img](./assets/clip_image020.jpg)

**4)** 测试模型；

在测试集上测试模型，并保存模型

![img](./assets/clip_image022.jpg)

**5)** 输出测试精度。

![img](./assets/clip_image024.jpg)

**1.** **以手写四则运算图片（白底黑字）为输入，完成：**

**1)** 使用连通域提取或者其它方法裁剪出图像中的每一个字符；

首先我们使用连通域提取方法，结果如下发现效果不是很理想，断点太多

![img](./assets/clip_image026.jpg)

![img](./assets/clip_image028.jpg)

因此我们换成了边缘检测切割

![img](./assets/clip_image030.jpg)

可以输出水平和垂直投影的直方图如下

![img](./assets/clip_image032.jpg)

然后可以通过投影来获取分割图像坐标

```python
def get_edge(test_img_path,img_name=None):
    """
        边缘提取
        :param img:图像矩阵
        :return:无
    """
    image=cv2.imread(test_img_path)
    #灰度化
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #二值化
    thresh=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,333,1)
    show_img(thresh,'thresh')
    
    #对图像垂直水平投影
    (h,w)=thresh.shape
    #垂直投影
    vproject=thresh.copy()
    a=[0 for z in range(0,w)]
    for j in range(0,w):#遍历一列
        for i in range(0,h):#遍历一行
            if vproject[i,j]==0:#如果改点为黑点
                a[j]+=1#该列的计数器加1计数
                vproject[i,j]=255#记录完后将其变为白色
    for j in range(0,w):#遍历每一列
        for i in range((h-a[j]),h):#从该列应该变黑的最顶部的点开始向最底部涂黑
            vproject[i,j]=0 #涂黑
    #水平投影
    hproject=thresh.copy()
    b=[0 for z in range(0,h)]#创建数组并初始化为0
    for i in range(0,h):#遍历一行
        for j in range(0,w):#遍历一列
            if hproject[i,j]==0:#如果改点为黑点
                b[i]+=1#该行的计数器加1计数
                hproject[i,j]=255#记录完后将其变为白色
    for i in range(0,h):#遍历每一行
        for j in range(0,b[i]):#从该行应该变黑的最左边的点开始向最右边涂黑
            hproject[i,j]=0 #涂黑
      
    plt.subplot(1,2,1)
    plt.imshow(vproject,cmap='gray')#显示图像
    plt.subplot(1,2,2)
    plt.imshow(hproject,cmap='gray')#显示图像
    plt.show()

    #分割字符
    th=thresh.copy()
    final_img=thresh.copy()
    show_img(final_img,'final_img')
    h_h = b#水平投影
    start = 0#用来记录起始位置
    h_start,h_end = [],[]#记录起始和终止位置
    position = []#记录分割位置
    #根据水平投影获取垂直分割
    for i in range(len(h_h)):
        if h_h[i] >0 and start==0:
            h_start.append(i)
            start=1
        if h_h[i] ==0 and start==1:
            h_end.append(i)
            start = 0
    for i in range(len(h_start)):
        cropImg = th[h_start[i]:h_end[i],0:w]#裁剪坐标为[y0:y1, x0:x1]
        if i==0:
            pass
        w_w = a
        wstart,wend,w_start,w_end = 0,0,0,0
        for j in range(len(w_w)):
            if w_w[j]>0 and  wstart==0:
                w_start = j
                wstart = 1
                wend = 0
            if w_w[j] ==0 and wstart==1:
                w_end = j
                wstart = 0
                wend = 1
            #当确定了起点和终点之后保存坐标
            if wend ==1:
                position.append([w_start,h_start[i],w_end,h_end[i]])
                wend = 0
    #根据坐标切割字符
    character_images = []
    for p in position:
        character_img = final_img[p[1]:p[3],p[0]:p[2]]
        character_img = cv2.equalizeHist(character_img)#直方图均衡化,为了增强对比度
        #阈值
        ret,character_img = cv2.threshold(character_img,254,255,cv2.THRESH_BINARY)
        character_images.append(character_img)
        # show_img(character_img)
    # save_img(character_images,img_name)
    return character_images
```

由于是从边缘切割，因此我们可以加上周围的留白，更好看一些

```python
def expand_img(character_images):
    expanded_imgs=[]
    width_max = 0
    hight_max = 0
    width_padding = 0
    hight_padding = 0
    offset=10
    for i,character_img in enumerate(character_images):
        width_max = character_img.shape[0]
        hight_max = character_img.shape[1]+10#加上偏移量因为这里图片高度固定，宽度不固定
        width_max = max(width_max,character_images[i].shape[1])+offset
        width_padding = width_max - character_images[i].shape[1]
        # hight_padding = hight_max - character_images[i].shape[0]
        expand_img = np.ones((width_max,width_max),dtype=np.uint8)*255
        #将原始图片放入扩展图片中心位置
        x_offset = width_padding//2
        y_offset = 0
        expand_img[y_offset:y_offset+character_images[i].shape[0],
                   x_offset:x_offset+character_images[i].shape[1]] = character_images[i]
        # expand_img[y_offset:y_offset+character_images[i].shape[0],x_offset:x_offset+character_images[i].shape[1]] = character_images[i]
        # character_images[i] = cv2.copyMakeBorder(character_images[i],0,0,0,padding,cv2.BORDER_CONSTANT,value=255)
        # show_img(character_images[i])
        # show_img(expand_img)
        expanded_imgs.append(expand_img)
    return expanded_imgs
```

可以看到边缘检测切割的效果很好

![img](./assets/clip_image040.jpg)

 

**2)** 调整字符图片大小为28*28像素；

写好预测函数，并将输入图片调整为28*28像素

```python

model.load_state_dict(torch.load('./lstm.ckpt'))
model.eval()
data_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28,28)),
    transforms.ToTensor()
])

#预测函数
def predict(img):
    #输出图片维度
    # print(img.size)
    img = data_transform(img).reshape(-1, 28, 28).to(device).float()
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs.data, 1)
        predicted = labels[predicted.item()]
        return predicted
```



**3)** 输入深度神经网络，识别图片中的所有字符，并输出识别结果；

```python
equation =[]
for img in character_images:
    predicted = predict(img)
    equation.append(predicted)
print(equation)
```

![img](./assets/clip_image046.jpg)

**4)** 输出图片中算式计算结果。

计算方法

```python
def calculate(equation):
    OPERATORS = {'plus': '+', 'sub': '-', 'mul': '*', 'div': '/'}
    # Transform equation to infix notation
    infix = []
    for item in equation:
        if item in OPERATORS:
            infix.append(OPERATORS[item])
        elif item == 'left':
            infix.append('(')
        elif item == 'right':
            infix.append(')')
        else:
            infix.append(item)
    infix_equation = ''.join(infix)

    # Evaluate the expression
    result = eval(infix_equation)

    return result, infix_equation
# equation = ['1', 'mul', '2', 'add', 'left','4','sub', '2','right']
result, infix_equation = calculate(equation)
print('equation:', equation)
print(infix_equation, '=', result)
```



**实例**： 

**输入图片：**

*![img](./assets/clip_image050.jpg)*

**输出结果：**

![img](./assets/clip_image052.jpg)
