import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def valid_training(model, loader, criterion, enable_gpu=False):
    correct = 0
    total = 0 
    if enable_gpu:
        model.to('cuda')
    else:
        model.to('cpu')
    running_val_loss = 0
    with torch.no_grad():
        for data in loader:
            imag, labs = data[0], data[1]
            if enable_gpu:
                imag, labs = data[0].to('cuda'),data[1].to('cuda')
            outputs = model.forward(imag)
            running_val_loss += criterion(outputs, labs)
            _, predicted = torch.max(outputs.data, 1)
            total += labs.size(0)
            correct += (predicted == labs).sum().item()
    print("Valid loss: {:.3f}".format(running_val_loss/len(loader)),
          "Accuracy: {:.3f}".format(100 * correct / total))
    
def predict_output(model, loader, dataset_type, criterion, enable_gpu=False):
    correct = 0
    total = 0
    if enable_gpu:
        model.to('cuda')
    else:
        model.to('cpu')
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data[0], data[1]
            if enable_gpu:
                images, labels = data[0].to('cuda'),data[1].to('cuda')
            outputs = model.forward(images)
            total_loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %s is: %d %%' % (dataset_type, 100 * correct / total))
    print("{} Loss: {:.4f}".format(dataset_type, total_loss / len(loader)))


def view_classify(img, ps, classes, k):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = np.flip(ps.data.numpy().squeeze(), axis=0)

    # Prepare image
    img = img.numpy().transpose((1, 2, 0))
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    img = np.clip(img, 0, 1)
    
    # Prepare predicted class name
    classes = classes.data.numpy().squeeze()
    names = [cat_to_name[str(clazz)] for clazz in classes][::-1]
    
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), nrows=2)
    ax1.imshow(img)
    ax1.axis('off')
    ax2.barh(np.arange(k), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(k))
    ax2.set_yticklabels(names, size='small')
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()    