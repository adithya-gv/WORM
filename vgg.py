import torch
import torchvision
import torch.nn as nn
import train
import time
import random

import torch.nn.utils.prune as prune

import sys

from earlyBird import EarlyBird
from agents import earlyBirdAgent, gradClipEBAgent, greedyClipEBAgent, wormEBAgent

def standard():
    model = torchvision.models.vgg11_bn(weights='DEFAULT', progress=True)



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

    prune_rate = 0.5

    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")
            prune.l1_unstructured(module, name="bias", amount=prune_rate)
            prune.remove(module, name="bias")
        if isinstance(module, nn.BatchNorm2d):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")



    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Test Model 
    accuracy = train.test(model, testloader, device)

    epoch = 0

    while accuracy < 90:
        # Train New Model until accuracy is 80%
        train.train_one_epoch(model, device, trainloader, optimizer, criterion, epoch)

        # Test New Model
        accuracy = train.test(model, testloader, device) 
        epoch += 1

    print(f"Training Time: {time.time() - start_time}")

    # env = earlyBirdRLAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, 10, device)
    # print(env.inference(0))

def gradClip():
    model = torchvision.models.vgg11_bn(weights='DEFAULT', progress=True)



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

    gradClipEBAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, 20, device, 0.1)

    prune_rate = 0.5

    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")
            prune.l1_unstructured(module, name="bias", amount=prune_rate)
            prune.remove(module, name="bias")
        if isinstance(module, nn.BatchNorm2d):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")



    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Test Model 
    accuracy = train.test(model, testloader, device)

    epoch = 0

    while accuracy < 90:
        # Train New Model until accuracy is 80%
        train.train_one_epoch(model, device, trainloader, optimizer, criterion, epoch)

        # Test New Model
        accuracy = train.test(model, testloader, device) 
        epoch += 1

    print(f"Training Time: {time.time() - start_time}")

    # env = earlyBirdRLAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, 10, device)
    # print(env.inference(0))

def greedyClip():
    model = torchvision.models.vgg11_bn(weights='DEFAULT', progress=True)



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

    greedyClipEBAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, 20, device)

    prune_rate = 0.5

    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")
            prune.l1_unstructured(module, name="bias", amount=prune_rate)
            prune.remove(module, name="bias")
        if isinstance(module, nn.BatchNorm2d):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")



    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Test Model 
    accuracy = train.test(model, testloader, device)

    epoch = 0

    while accuracy < 90:
        # Train New Model until accuracy is 80%
        train.train_one_epoch(model, device, trainloader, optimizer, criterion, epoch)

        # Test New Model
        accuracy = train.test(model, testloader, device) 
        epoch += 1

    print(f"Training Time: {time.time() - start_time}")

    # env = earlyBirdRLAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, 10, device)
    # print(env.inference(0))

def worm():
    model = torchvision.models.vgg11_bn(weights='DEFAULT', progress=True)



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

    wormEBAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, 20, device)

    prune_rate = 0.5

    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")
            prune.l1_unstructured(module, name="bias", amount=prune_rate)
            prune.remove(module, name="bias")
        if isinstance(module, nn.BatchNorm2d):
            prune.l1_unstructured(module, name="weight", amount=prune_rate)
            prune.remove(module, name="weight")



    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # Test Model 
    accuracy = train.test(model, testloader, device)

    epoch = 0

    while accuracy < 90:
        # Train New Model until accuracy is 80%
        train.train_one_epoch(model, device, trainloader, optimizer, criterion, epoch)

        # Test New Model
        accuracy = train.test(model, testloader, device) 
        epoch += 1

    print(f"Training Time: {time.time() - start_time}")

    # env = earlyBirdRLAgent(model, criterion, optimizer, trainloader, testloader, earlyBird, 10, device)
    # print(env.inference(0))

if __name__ == "__main__":
    # Init random seed for torch
    torch.manual_seed(random.randint(0, 1e5))
    if sys.argv[1] == "standard":
        standard()
    elif sys.argv[1] == "gradclip":
        gradClip()
    elif sys.argv[1] == "greedy":
        greedyClip()
    elif sys.argv[1] == "worm":
        worm()
    else:
        print("Invalid Argument")
        exit(1)
