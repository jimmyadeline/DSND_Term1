import argparse
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict
from helper import predict_output
from enum import Enum

from helper import valid_training

#    Basic usage: python train.py data_directory
#    Prints out training loss, validation loss, and validation accuracy as the network trains
#    Options:
#        Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
#        Choose architecture: python train.py data_dir --arch "vgg16"
#        Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
#        Use GPU for training: python train.py data_dir --gpu

class Arch(Enum):
    densenet121 = 'densenet121'
    vgg16 = 'vgg16'
    def __str__(self):
        return self.value

parser = argparse.ArgumentParser(description="Prints out training loss, validation loss, and validation accuracy as the network trains")
parser.add_argument("data_dir", help="path of the data directory")
parser.add_argument("--save_dir", help="path of the save directory")
parser.add_argument("--arch", help="the architecture of the pretrained network", type=Arch, choices=list(Arch))
parser.add_argument("--learning_rate", help="learning rate", type=float)
parser.add_argument("--hidden_units", help="hidden unit", type=int)
parser.add_argument("--epochs", help="epochs", type=int)

parser.add_argument("--gpu", help="enable gpu", action="store_true")

args = parser.parse_args()


def load_dataset(data_dir):
    """
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'    
    # Define transforms for the training, validation, and testing sets 
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    return train_data, valid_data, test_data, trainloader, validloader, testloader
  
# Building and training the classifier¶
def build_model(arch, hidden_units):
    """
    """
    # Load a pre-trained network    
    model = getattr(models, arch)(pretrained=True) ## model = models.densenet121(pretrained=True)
    # Define a new, untrained feed-forward network as a classifier, use ReLU activations and dropout
    for param in model.parameters():
        param.requires_grad = False

    last_layer_size = 1024    # densenet121
    if arch == "vgg16":
        last_layer_size = 25088
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(last_layer_size, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier 
    return model

def train_model(criterion, optimizer, model, epochs, trainloader):
#    # Train the classifier layers using backpropagation using the pre-trained network to get the features
#     criterion = nn.NLLLoss()
#     optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    print_every = 40
    steps = 0

    # change to cuda
    if args.gpu:
        model.to('cuda')
    else:
        model.to('cpu')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            
            if args.gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))
                model.eval()
                valid_training(model, validloader, criterion, args.gpu)
                model.train()
                
                running_loss = 0

# Driver
##### Default parameter setting
data_dir = args.data_dir
arch = "densenet121"
hidden_units = 400
learning_rate = 0.001
epochs = 3
save_dir = ''
if args.arch:
    arch = str(args.arch)
    
if args.save_dir:
    save_dir = args.save_dir + '/'
if args.learning_rate:
    learning_rate = args.learning_rate
if args.hidden_units:
    hidden_units = args.hidden_unit
if args.epochs:
    epochs = args.epochs
if args.gpu:
    print("enable gpu")
    
    
train_data, valid_data, test_data, trainloader, validloader, testloader = load_dataset(data_dir)
model = build_model(arch, hidden_units)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
train_model(criterion, optimizer, model, epochs, trainloader)
predict_output(model, trainloader, "training set", criterion, args.gpu)
predict_output(model, validloader, "validation set", criterion, args.gpu)
predict_output(model, testloader, "test set", criterion, args.gpu)

def save_checkpoint(arch, learning_rate, model, epochs, optimizer, criterion, train_data):
    checkpoint = {'arch': str(arch),
                  'learning_rate': learning_rate,
                  'epochs': epochs,
                  'classifier': model.classifier,
                  'optimizer_state': optimizer.state_dict(),
                  'model_state': model.state_dict(),
                  'criterion_state': criterion.state_dict(),
                  'class_to_idx': train_data.class_to_idx
                 }
    torch.save(checkpoint, '%scheckpoint_%s.pth' % (save_dir, arch))

save_checkpoint(str(arch), learning_rate, model, epochs, optimizer, criterion, train_data)

