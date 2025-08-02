# This file is based on the `https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html`.

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pdb
from core.hme_seq import HierarchicalMoE
from utils.config import MoEConfig
from core.hierarchical_moe import MoE


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.soft(out)
        return out


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


if torch.cuda.is_available():
    device = torch.device('cuda:3')
else:
    device = torch.device('cpu')

config = MoEConfig(
    num_experts=(3, 5),
    moe_input_size=3072,
    moe_hidden_size=256,
    moe_output_size=10,
    top_k=(2, 4),
    router_type='joint',
    gating=('softmax' ,'softmax'),
    noisy_gating=False
)

net = HierarchicalMoE(config)

# net = HierarchicalMoE(
#     input_dim = 3072,
#     output_dim = 10,
#     hidden_dim = 256,
#     num_experts = (4, 4)
# )

# net = MoE(
#     input_dim = 3072,
#     output_dim = 10,
#     hidden_dim = 256,
#     num_experts = 16
# )

net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95)
# optimizer = optim.Adam(net.parameters())

net.train()
for epoch in range(200):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        pdb.set_trace()
        inputs = inputs.view(inputs.shape[0], -1) # check this!
        outputs, aux_loss = net(inputs)
        # outputs, aux_loss = net(inputs)
        # outputs = net(inputs)
        loss = criterion(outputs.squeeze(), labels)
        # total_loss = loss + aux_loss
        # total_loss.backward()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')


correct = 0
total = 0
net.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs, _ = net(images.view(images.shape[0], -1))
        # outputs = net(images.view(images.shape[0], -1))
        _, predicted = torch.max(outputs.squeeze().data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

# yields a test accuracy of around 34 %
