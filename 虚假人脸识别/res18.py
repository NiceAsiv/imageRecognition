# 数据集划分.py
# 给定一个人脸数据集，其中包含1999张真实人脸，1999张虚假人脸。
# 将其中500张真实人脸和500张虚假人脸作为训练集，其余作为测试集。
from __future__ import print_function, division
import os, random, shutil


def eachFile(filepath):
    pathDir = os.listdir(filepath)
    return pathDir

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def divideTrainValiTest(source, dist):
    print("开始划分数据集...")
    print(eachFile(source))
    for c in eachFile(source):
        pic_name = eachFile(os.path.join(source, c))
        random.shuffle(pic_name)  # 随机打乱
        train_list = pic_name[0:1499]
        validation_list = pic_name[1499:]
        test_list = []
        
        mkdir(dist+ 'train/'+c+'/')
        mkdir(dist+ 'validation/'+c+'/')
        mkdir(dist+ 'test/'+c+'/')
        
        for train_pic in train_list:
            shutil.copy(os.path.join(source, c, train_pic),
                        dist+ 'train/'+c+'/'+train_pic)

        for validation_pic in validation_list:
            shutil.copy(os.path.join(source, c, validation_pic),
                        dist+ 'validation/'+c+'/'+validation_pic)

        for test_pic in test_list:
            shutil.copy(os.path.join(source, c, test_pic),
                        dist + 'test/'+c+'/'+test_pic)
    return


if __name__ == '__main__':
    filepath = r'./CNN_synth_testset/'
    dist = r'./CNN_synth_testset_divided/'
    divideTrainValiTest(filepath, dist)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True#加速卷积神经网络的训练，但是每次的结果可能不同
plt.ion()   # interactive mode

# Data augmentation and normalization for training
# Just normalization for validation
# https://blog.csdn.net/KaelCui/article/details/106175313
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'validation': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
batch_size = 4
data_dir = './CNN_synth_testset_divided/'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'validation']}#数据集的读取
print(image_datasets)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=0) 
              for x in ['train', 'validation']}#数据集的读取
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
def my_imgs_plot(image, labels, preds=None):
    cnt = 0
    plt.figure(figsize=(16,16)) 
    for j in range(len(image)):
        cnt += 1
        plt.subplot(1,len(image),cnt)
        plt.xticks([], [])
        plt.yticks([], [])
        if preds is not None:
            plt.title(f"predicted: {class_names[preds[j]]}, true: {class_names[labels[j]]}\n"
                      ,color='green' if preds[j] == labels[j] else 'red')
        else:
            plt.title("{}".format(class_names[labels[j]]), fontsize=15, color='green')
        inp = image[j].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
    plt.show()

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))#读取一个batch的数据

my_imgs_plot(inputs, classes)

from tqdm import tqdm
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    loss_plot = []
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs)):
        print(f'Epoch {epoch}/{num_epochs - 1}:',end=' ')

        # 每个 epoch 都有一个培训和验证阶段
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # 将模型设置为训练模式
            else:
                model.eval()  # 将模型设置为评估模式

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据。
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 将参数梯度归零
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            loss_plot.append(epoch_loss)
            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    plt.plot(loss_plot)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

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

            my_imgs_plot(inputs.cpu(),labels, preds)
            images_so_far += batch_size
            # for j in range(inputs.size()[0]):
            #     plt.figure(figsize=(12,12))
            #     images_so_far += 1
            #     ax = plt.subplot(num_images//2, 2, images_so_far)
            #     ax.axis('off')
            #     ax.set_title(f"predicted: {class_names[preds[j]]}, true: {class_names[labels[j]]},{'success' if preds[j] == labels[j] else 'failure'}")
            #     imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                model.train(mode=was_training)
                return
        model.train(mode=was_training)

model_conv = torchvision.models.resnet18(pretrained=True)
for param in model_conv.parameters():
    param.requires_grad = False#冻结参数

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# 注意只有最后一层的参数正在优化，而不是之前。
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=25)
PATH = './fakeFace_tf_{}.pth'.format(model_conv.__class__.__name__)
torch.save(model_conv.state_dict(), PATH) 

visualize_model(model_conv)

plt.ioff()
plt.show()
