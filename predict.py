
  
# Imports
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import matplotlib as mpl
from matplotlib.pyplot import plot, draw, show
matplotlib.use('Agg')
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)
import argparse
import numpy as np
import pandas as pd

import torch
import seaborn as sb

from torch import nn
from torch import optim
import torch.nn.functional as F

from torchvision import datasets, transforms, models
#import torchvision
#import torchvision.models as models
from PIL import Image
import json

def main():
    
    # get arguments from command line
    input = get_args()
    
    path_to_image = input.image_path
    checkpt = input.checkpoint
    num = input.top_k
    cat_names = input.category_names
    gpu = input.gpu
    
    # load category names file
    with open(cat_names, 'r') as f:
        cat_to_name = json.load(f)
        
    # load trained model
    model = load(checkpt)
    
    # Process images, predict classes, and display results
    #img = Image.open(path_to_image)
    image = process_image(path_to_image)
    probs, classes = predict(path_to_image, model)
    check(image,path_to_image,model,cat_to_name)
    
    
    

# Function Definitions
def get_args():
    """
        Get arguments from command line
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--image_path", type=str, help="path to image in which to predict class label")
    parser.add_argument("--checkpoint", type=str, help="checkpoint in which trained model is contained")
    parser.add_argument("--top_k", type=int, default=5, help="number of classes to predict")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json",
                        help="file to convert label index to label names")
    parser.add_argument("--gpu", type=bool, default=True,
                        help="use GPU or CPU to train model: True = GPU, False = CPU")
    
    return parser.parse_args()

# define NeuralNetwork Class with FeedForward Method
class NeuralNetwork(nn.Module):
    # define layers of the neural network: input, output, hidden layers
    def __init__(self, input_size, output_size, hidden_layers):

        # calls init method of nn.Module (base class)
        super().__init__()

        # input_size to hidden_layer 1 : ModuleList --> list meant to store nn.Modules
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # add arbitrary number of hidden layers
        i = 0
        j = len(hidden_layers)-1

        while i != j:
            l = [hidden_layers[i], hidden_layers[i+1]]
            self.hidden_layers.append(nn.Linear(l[0], l[1]))
            i+=1

        # check to make sure hidden layers formatted correctly
        for each in hidden_layers:
            print(each)

        # last hidden layter -> output
        self.output = nn.Linear(hidden_layers[j], output_size)

    # feedforward method    
    def forward(self, tensor):

        # Feedforward through network using relu activation function
        for linear in self.hidden_layers:
            tensor = F.relu(linear(tensor))
        tensor = self.output(tensor)

        # log_softmax: better for precision (numbers not close to 0, 1)
        return F.log_softmax(tensor, dim=1)
    
    
def load(x):
    """
        Load the saved trained model inorder to use for prediction
    """
    checkpoint = torch.load(x)
    #model = getattr(models, checkpoint['arch'])(pretrained=True)
    
   
    if checkpoint ['arch'] == 'alexnet':
        model = models.alexnet (pretrained = True)
    else: #vgg13 as only 2 options available
        model = models.vgg16 (pretrained = True)    
    classifier = NeuralNetwork(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'],
                             )
    model.classifier = classifier

    
    model.classifier.load_state_dict(checkpoint['state_dict'])
    
    model.classifier.optimizer = checkpoint['optimizer']
    model.classifier.epochs = checkpoint['epochs']
    model.classifier.learning_rate = checkpoint['learning_rate']
    
    for param in model.parameters():
        param.requires_grad = False
    return model

def process_image(path_to_image):
    ''' 
        Transform an image so model can successfully predict its class.
    '''
    # resize and crop image
    img = Image.open(path_to_image)
    prepoceess_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
    # TODO: Process a PIL image for use in a PyTorch model
    pymodel_img = prepoceess_img(img)
    return pymodel_img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.array (image)
    image = image.transpose((1, 2, 0))
    
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(path_to_image, model, topk=5):   
    model.to('cuda:0')
    img_torch = process_image(path_to_image)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)
    


    
def check(image,path_to_image, model,cat_to_name):
    """
        Ouput a picture of the image and a graph representing its top 'k' class labels
    """
#path_to_image = 'flowers/test/11/image_03115.jpg'
#image=process_image(path_to_image)
    probs = predict(path_to_image, model)
    probs=probs
    a = np.array(probs[0][0])
    b = [cat_to_name[str(index + 1)] for index in np.array(probs[1][0])]
    sb.barplot(y = a, x = b, color ='blue', ecolor='black')

    show()
    imshow(image)
    print(a,b)
    
# Run the program
if __name__ == "__main__":
    main()


