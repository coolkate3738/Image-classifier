
  
# Imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable
import argparse
import numpy as np
import pandas as pd
import torch
import seaborn as sb
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json


parser = argparse.ArgumentParser()
parser.add_argument('--image_path', action="store",default='./flowers/valid/10/image_07094.jpg' )
parser.add_argument('--checkpoint_path', action="store",default='./my_checkpoint.pth')
parser.add_argument('--category_names_mapping', action="store", default= './cat_to_name.json')
parser.add_argument ('--gpu', dest="gpu", action="store",choices=['gpu', 'cpu'],type = str,default='gpu')
parser.add_argument ('--top_n',  action="store", type = int, default=5)
parsed_args = parser.parse_args()


parsed_args = parser.parse_args()
if parsed_args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'


def transform_image(image):
    img = Image.open(image)

    transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    transformed_img = transform(img)
    transformed_img_array = np.array(transformed_img)
    
    # Converting to torch tensor from Numpy array
    tensor = torch.from_numpy(transformed_img_array).type(torch.FloatTensor)
    
    tensor_mod = tensor.unsqueeze_(0)
    return tensor_mod


def predict(image_tensor, model,device,top_n):
    
    model.eval()
    if device == 'cuda':
        model.to('cuda')

    else:
        model.cpu()
    
    with torch.no_grad():
        # Running image through network
        output = model.forward(image_tensor)

    # Calculating probabilities
    probs = torch.exp(output)
    top_probs = probs.topk(top_n)[0]
    top_labels = probs.topk(top_n)[1]
    
    # Converting probabilities and outputs to lists
    probs_list = np.array(top_probs)[0]
    index_list = np.array(top_labels[0])
    
    # Loading index and class mapping
    class_to_idx = model.class_to_idx
    # Inverting index-class dictionary
    indx_to_class = {x: y for y, x in class_to_idx.items()}

    # Converting index list to class list
    classes_list = []
    for index in index_list:
        classes_list += [indx_to_class[index]]
        
    return probs_list, classes_list


def load_checkpoint(filepath):
    checkpoint = torch.load (filepath) #loading checkpoint from a file
    if checkpoint ['arch'] == 'vgg16':
        model = models.vgg16 (pretrained = True)
    else: #vgg19 
        model = models.vgg19 (pretrained = True)
        checkpoint = torch.load(filepath)
    
    #freezing the parameters
    for param in model.parameters():
        param.requires_grad = False
    
   

    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.optimizer_state = checkpoint['optimizer_state']
    model.epochs = checkpoint['epochs']
    return model


        
def main():


    # Load categories to names json file
    if parsed_args.gpu == 'gpu':
        device = 'cuda'
    else:
        device = 'cpu'
    with open(parsed_args.category_names_mapping, 'r') as f:
        cat_to_name = json.load(f)
    # Load model trained with train.py
    model = load_checkpoint(parsed_args.checkpoint_path)        
    image_tensor = transform_image(parsed_args.image_path)        
    

    if device == 'cuda':
        image_tensor = image_tensor.to('cuda')
    else:
        pass
    top_probs, top_classes= predict(image_tensor, model,device, top_n=parsed_args.top_n)
    decoded_names = []
    for i in top_classes:
        decoded_names += [cat_to_name[i]]


    print(f"most likely this flower is: '{decoded_names[0]}' with {(top_probs[0]*100)}% ")


if __name__ == '__main__': main()        