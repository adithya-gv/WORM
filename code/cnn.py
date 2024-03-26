import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import train
import time

import numpy as np

from earlyBird import EarlyBird
from agents import earlyBirdAgent, earlyBirdRLAgent, fasterEarlyBirdAgent

class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
	    """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
		"""
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,)) 
        output = input_tensor[:, selected_index, :, :]
        return output

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.select = channel_selection(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 10)

    def _make_layer(self, in_channels, planes, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, 1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.select(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)
        
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, planes, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, planes, 3, stride, padding=1),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        residual = self.shortcut(residual)
        x += residual
        return F.relu(x)


model = ResNet18()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Get CIFAR-10 Dataset
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Init EarlyBird package, with threshold parameter of 0.1
earlyBird = EarlyBird(0.5, 5, 0.1)

start_time = time.time()

earlyBirdAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, 20, device)
mask = earlyBird.pruning(model)
model = earlyBird.final_prune(model, mask, device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Test Model 
accuracy = train.test(model, testloader, device)

epoch = 0

while accuracy < 90:
    # Train New Model until accuracy is 90%
    train.train_one_epoch(model, device, trainloader, optimizer, criterion, epoch)

    # Test New Model
    accuracy = train.test(model, testloader, device) 
    epoch += 1

print(f"Training Time: {time.time() - start_time}")

# env = earlyBirdRLAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, 10, device)
# print(env.inference(0))