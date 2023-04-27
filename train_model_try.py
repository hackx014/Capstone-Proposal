#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
# Copy of hpo_NH2.py file.
import numpy as np
import torch
import argparse
import smdebug.pytorch as smd
import os
import json
import logging
import sys
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import models, transforms
from torch import nn, optim
from PIL import ImageFile


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# Define the network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # fix the input shape
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)  # fix the input shape
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, train_loader, criterion, optimizer, epoch, hook):
    model.train()
    hook.set_mode(smd.modes.TRAIN)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def test(model, test_loader, criterion, hook):
    model.eval()
    hook.set_mode(smd.modes.EVAL)
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args, _ = parser.parse_known_args()

    # Define the transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Load the data
    train_dataset = ImageFolder(root=args.data_dir , transform=transform)
    test_dataset = ImageFolder(root=args.data_dir , transform=transform)

    # Create the data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize the model
    model = Net()
    hook = smd.Hook.create_from_json_file()
    hook.register_module(model)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    hook.register_loss(criterion)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    # Train and test the model
    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, criterion, optimizer, epoch, hook)
        test(model, test_loader, criterion, hook)

    # Save the model
    torch.save(model.state_dict(), os.path.join(os.environ['SM_MODEL_DIR'], 'model.pth'))

if __name__ == '__main__':
    main()

