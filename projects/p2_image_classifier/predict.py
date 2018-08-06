import json
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from collections import OrderedDict

#    Basic usage: python predict.py /path/to/image checkpoint
#    Options:
#        Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
#        Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#        Use GPU for inference: python predict.py input checkpoint --gpu
parser = argparse.ArgumentParser(description="Prints out checkpoint")
parser.add_argument("input", help="path of input image file")
parser.add_argument("checkpoint", help="path of checkpoint *.pth file")
parser.add_argument("--top_k", help="return top most likely classes")
parser.add_argument("--category_names", help="use a mapping of categories to real names")
parser.add_argument("--gpu", help="enable gpu", action="store_true")

args = parser.parse_args()

if args.top_k:
    print(args.top_k)
if args.gpu:
    print("enable gpu")

# label mapping
cat_to_name = {}
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.densenet121(pretrained=True)
#     model = getattr(models, arch)(pretrained=True)
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['model_state'])
#     optimizer.load_state_dict(checkpoint['optimizer_state'])
#     criterion.load_state_dict(checkpoint['criterion_state'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model    

def predict_output(model, loader, dataset_type):
    correct = 0
    total = 0
    if args.gpu:
        model.to('cuda')
    model.eval()
    with torch.no_grad():
        for data in loader:
            if args.gpu:
                images, labels = data[0].to('cuda'),data[1].to('cuda')
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %s is: %d %%' % (dataset_type, 100 * correct / total))

# Inference
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = Image.open(image)
    
    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    
    img_processed = transformations(img)
    
    return img_processed
    
def predict(image_path, model, k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()  ### set the model in inference mode

    img = process_image(image_path)
    print(img.shape)
    img = torch.from_numpy(np.expand_dims(img, axis=0)).to('cuda')

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img)

    ps = torch.exp(output)
    
    probs, indices = ps.topk(k)
    idx_to_class = {v:k for (k,v) in model.class_to_idx.items()}
    get_classes = np.vectorize(lambda x : int(idx_to_class[x]))
    classes_np = get_classes(indices.cpu().numpy().squeeze())
    classes = torch.from_numpy(classes_np)
    return probs, classes

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

    
learning_rate = 0.001
criterion = nn.NLLLoss()
checkpoint_path = args.checkpoint
model = load_checkpoint(checkpoint_path)
if args.gpu:
    model.to('cuda')
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# driver
image_path = args.input
image = process_image(image_path)
topk = 5
if args.top_k:
    topk = args.top_k
probs, classes = predict(image_path, model, topk)
if args.gpu:
    probs = probs.cpu()
    classes = classes.cpu()
print(probs)
print(classes)
# view_classify(image, probs, classes, topk)

