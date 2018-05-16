# from IPython import get_ipython
# get_ipython().run_line_magic('matplotlib', 'inline')
# HOW TO RUN:  python p1.py
#  To use different LEARNING RATES: change line 169
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import itertools
# import seaborn as sns

cuda = torch.cuda.is_available()
print('Using PyTorch version:', torch.__version__, 'CUDA:', cuda)

#torch.manual_seed(42)
#if cuda:
#    torch.cuda.manual_seed(42)
batch_size = 32

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=batch_size, shuffle=True, **kwargs)

validation_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=batch_size, shuffle=False, **kwargs)

print validation_loader

for (X_train, y_train) in train_loader:
    print('X_train:', X_train.size(), 'type:', X_train.type())
    print('y_train:', y_train.size(), 'type:', y_train.type())
    break

pltsize=1
plt.figure(figsize=(10*pltsize, pltsize))

#for i in range(10):
#    plt.subplot(1,10,i+1)
#    plt.axis('off')
#    plt.imshow(X_train[i,:,:,:].numpy().reshape(32,32), cmap="rgb")
#    plt.title('Class: '+str(y_train[i]))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #self.conv1 = nn.Conv2d(3, 3, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc1_drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(100, 10)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32*32*3)
        x = self.sig(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if cuda:
    model.cuda()

# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

print(model)

def train(epoch, optimizer, log_interval=100):

    # end position of training set (first 4 batches AKA first 4/5s of the training set)
    trainingEndIndex = int(len(train_loader)*.8)

    # print "\nlength of train_loader: ", len(train_loader)
    # print "\nlength of validation_loader: ", len(validation_loader)

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), 40000, # was len(train_loader.dataset)
                100. * batch_idx / (len(train_loader)*.8), loss.data[0]))

        # only train on the first 40k samples (batch 1 - 4)
        if batch_idx >= trainingEndIndex:
            break

def validate(loss_vector, accuracy_vector):
    # print "\nlength of train_loader: ", len(train_loader)
    # print "\nlength of validation_loader: ", len(validation_loader)

    # start position of validation set (last batch AKA last 4/5s of the training set)
    validationStartingIndex = int(len(train_loader)*.8)
    model.eval()
    val_loss, correct = 0, 0
    sampleCount = 0
    # for data, target in validation_loader:
    for sample_idx, (data, target) in enumerate(train_loader):
        # print sample_idx, " ", sampleCount, "    "
        if sample_idx > validationStartingIndex:
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            val_loss += F.nll_loss(output, target).data[0]
            pred = output.data.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            sampleCount += 1

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct / 10000 # was len(validation_loader.dataset) 
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, 10000, accuracy)) # was len(validation_loader.dataset) 

def test(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        val_loss += F.nll_loss(output, target).data[0]
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    # loss_vector.append(val_loss)

    accuracy = 100. * correct / len(validation_loader.dataset)
    # accuracy_vector.append(accuracy)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))

# %%time
epochs = 10
learningRates = [0.0001]
lossvE, accvE = [], []
for i in learningRates:
    optimizer = optim.SGD(model.parameters(), lr=i, momentum=0.5)
    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        train(epoch, optimizer)
        validate(lossv, accv)
    lossvE.append(lossv)
    accvE.append(accv)
lossv, accv = [], []
test(lossv, accv)
plt.figure(figsize=(10,8))
plt.ylabel('Negative Log Loss')
plt.xlabel('Epochs')
# plt.plot(np.arange(1,epochs+1), lossvE[0], color="cyan", label='LR: 0.1')
# plt.plot(np.arange(1,epochs+1), lossvE[0], color="red", label='LR: 0.01')
# plt.plot(np.arange(1,epochs+1), lossvE[0], color="green", label='LR: 0.001')
plt.plot(np.arange(1,epochs+1), lossvE[0], color="blue", label='LR: 0.0001')
plt.title('Average Loss')
plt.legend()
plt.tight_layout()
plt.savefig('p1partLoss0001.png')

plt.figure(figsize=(10,8))
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
# plt.plot(np.arange(1,epochs+1), accvE[0], color="cyan", label='LR: 0.1')
# plt.plot(np.arange(1,epochs+1), accvE[0], color="red", label='LR: 0.01')
# plt.plot(np.arange(1,epochs+1), accvE[0], color="green", label='LR: 0.001')
plt.plot(np.arange(1,epochs+1), accvE[0], color="blue", label='LR: 0.0001')
plt.title('Validation Accuracy');
plt.legend()
plt.tight_layout()
plt.savefig('p1partValidation0001.png')