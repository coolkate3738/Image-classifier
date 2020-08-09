
  
# Imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
parser.add_argument('--image_path', default='./flowers/valid/10/image_07094.jpg' )
parser.add_argument('--checkpoint_path', default='./my_checkpoint.pth')
parser.add_argument('--category_names_mapping',  default= './cat_to_name.json')
parsed_args = parser.parse_args()


def transform_image(image):
    img = Image.open(image)

    transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    transformed_img = transform(img)
    return np.array(transformed_img)


def predict(imagepath, model):
    #imagepath= data_dir+'/test' + "/80/image_01983.jpg"
    top_n = 6
    model.eval()
    ransformed_image = transform_image(imagepath)
    torch_image = torch.from_numpy(np.expand_dims(transform_image(imagepath), 
                                                      axis=0)).type(torch.FloatTensor).to("cpu")
    torch_image.to('cpu')
    model.to('cpu')


    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        outputs = model(torch_image)
        logarithmic = model.forward(torch_image)
        
        probs = torch.exp(logarithmic)

        
        top_probs, top_labels = probs.topk(top_n)

        top_probs = top_probs.numpy()[0]
        top_labels = top_labels.numpy()[0]
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_labels = [idx_to_class[i] for i in top_labels]
    return top_probs, top_labels

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True);
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
    with open(parsed_args.category_names_mapping, 'r') as f:
        cat_to_name = json.load(f)
    # Load model trained with train.py
    model = load_checkpoint(parsed_args.checkpoint_path)        
    image_tensor = transform_image(parsed_args.image_path)        
    top_probs, top_labels = predict(parsed_args.image_path, model)       
    print(top_probs, top_labels)


if __name__ == '__main__': main()        