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
from helper import predict_output

#    Basic usage: python predict.py /path/to/image checkpoint
#    Options:
#        Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
#        Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
#        Use GPU for inference: python predict.py input checkpoint --gpu
parser = argparse.ArgumentParser(description="Prints out checkpoint")
parser.add_argument("input", help="path of input image file")
parser.add_argument("checkpoint", help="path of checkpoint *.pth file")
parser.add_argument("--top_k", help="return top most likely classes", type=int)
parser.add_argument("--category_names", help="use a mapping of categories to real names")
parser.add_argument("--gpu", help="enable gpu", action="store_true")

args = parser.parse_args()

if args.gpu:
    print("enable gpu")

# label mapping
cat_to_name = {}
if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    arch = checkpoint['arch']
#     arch = 'vgg16'
    model = getattr(models, arch)(pretrained=True)
    model.classifier = checkpoint['classifier']
    epochs = checkpoint['epochs']
    model.load_state_dict(checkpoint['model_state'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model    

# Inference
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    
    transformations = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
    
    img_processed = transformations(img)
    
    return img_processed
    
def predict(image_path, model, k=5, enable_gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()  ### set the model in inference mode

    img = process_image(image_path)
    img = torch.from_numpy(np.expand_dims(img, axis=0))
    if enable_gpu:
        img = img.to('cuda')

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

    
checkpoint_path = args.checkpoint
model = load_checkpoint(checkpoint_path)
if args.gpu:
    model.to('cuda')
else:
    model.to('cpu')

# driver
image_path = args.input
image = process_image(image_path)
topk = 1
if args.top_k:
    topk = args.top_k
probs, classes = predict(image_path, model, topk, args.gpu)
if args.gpu:
    probs = probs.cpu()
    classes = classes.cpu()
print(probs)
print(classes)

if args.category_names:
    classes = classes.data.numpy()
    if classes.shape != ():
        names = [cat_to_name[str(clazz)] for clazz in classes]
        print(names)
    else:
        names = cat_to_name[str(classes)] # size 1
        print(names)
    
