"""
---------------------------------------------------------------------
Training an image classifier
---------------------------------------------------------------------
For this assignment you'll do the following steps in order:
1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolutional Neural Network (at least 4 conv layer)
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data
---------------------------------------------------------------------
"""

# IMPORTING REQUIRED PACKAGES
import os
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from cnn_model import ConvNet

# DEFINE VARIABLE
BATCH_SIZE = 128  # YOU MAY CHANGE THIS VALUE
EPOCH_NUM = 25  # YOU MAY CHANGE THIS VALUE
LR = 0.001  # YOU MAY CHANGE THIS VALUE
MODEL_SAVE_PATH = './Models'

if not os.path.exists(MODEL_SAVE_PATH):
    os.mkdir(MODEL_SAVE_PATH)

# DEFINING TRANSFORM TO APPLY TO THE IMAGES
# YOU MAY ADD OTHER TRANSFORMS FOR DATA AUGMENTATION
transform = transforms.Compose(
    [transforms.Resize(32),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

########################################################################
# 1. LOAD AND NORMALIZE CIFAR10 DATASET
########################################################################

# FILL IN: Get train and test dataset from torchvision and create respective dataloader
trainset = CIFAR10('./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

testset = CIFAR10('./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True)

########################################################################
# 2. DEFINE YOUR CONVOLUTIONAL NEURAL NETWORK AND IMPORT IT
########################################################################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = ConvNet().to(device)  # MAKE SURE TO DEFINE ConvNet IN A CELL ABOVE THE STARTER CODE OF WHICH IS IN cnn_model.py
# You can pass arguments to ConvNet if you want instead of hard coding them.


########################################################################
# 3. DEFINE A LOSS FUNCTION AND OPTIMIZER
########################################################################

# FILL IN : the criteria for ce loss
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

########################################################################
# 4. TRAIN THE NETWORK
########################################################################

test_accuracy = []
train_accuracy = []
train_loss = []
test_min_acc = 0
net.train()

for epoch in range(EPOCH_NUM):  # loop over the dataset multiple times

    running_loss = 0.0
    total = 0
    correct = 0

    for i, data in enumerate(trainloader, 0):
        net.train()
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)

        # FILL IN: Obtain accuracy for the given batch of TRAINING data using
        # the formula acc = 100.0 * correct / total where 
        # total is the total number of images processed so far
        # correct is the correctly classified images so far

        total += inputs.size(0)
        correct += torch.count_nonzero((predicted == labels).long()).item()

        train_loss.append(loss.item())
        train_accuracy.append(100.0 * correct / total)

        if (i + 1) % 20 == 0:
            print('Train: [%d, %5d] loss: %.3f acc: %.3f' % (epoch + 1, i + 1,
                                                             running_loss / 20,
                                                             100.0 * correct / total))
            running_loss = 0.0

    # TEST LEARNT MODEL ON ENTIRE TESTSET
    # FILL IN: to get test accuracy on the entire testset and append 
    # it to the list test_accuracy

    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            # YOUR CODE HERE
            test_inputs, test_labels = data
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_preds = net(test_inputs)
            test_preds = torch.argmax(test_preds, dim=1)
            total += test_inputs.size(0)
            correct += torch.count_nonzero((test_preds == test_labels).long()).item()
        test_accuracy.append(100.0 * correct / total)
    net.train()

    test_ep_acc = test_accuracy[-1]
    print('Test Accuracy: %.3f %%' % test_ep_acc)

    # SAVE BEST MODEL
    if test_min_acc < test_ep_acc:
        test_min_acc = test_ep_acc
        torch.save(net, MODEL_SAVE_PATH + '/my_best_model.pth')

# PLOT THE TRAINING LOSS VS EPOCH GRAPH
plt.figure(figsize=(10, 5))
plt.plot(list(range(len(train_loss))), train_loss)
plt.title('training loss vs. iteration')
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.savefig('training_loss_vs_iteration_plot.jpg', dpi=300)
plt.show()

# PLOT THE TESTING ACCURACY VS EPOCH GRAPH
plt.figure(figsize=(10, 5))
plt.plot(list(range(len(test_accuracy))), test_accuracy)
plt.title('test accuracy vs. epoch')
plt.xlabel('epoch')
plt.ylabel('test_accuracy')
plt.savefig('test_accuracy_vs_epoch_plot.jpg', dpi=300)
plt.show()

# PRINT THE FINAL TESTING ACCURACY OF YOUR BEST MODEL
print('the best testing accuracy: {}'.format(max(test_accuracy)))

print('Finished Training')
