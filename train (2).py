import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json

parser = argparse.ArgumentParser()
parser.add_argument('--arch', dest='arch', action='store', choices=['vgg16', 'vgg19'], default='vgg16', help='Choose from vgg16 or vgg19')
parser.add_argument('--save_dir', dest="save_dir", action="store", default=".")
parser.add_argument('--learning_rate', dest="learning_rate", action="store",type=float, default=0.001)
parser.add_argument ('--hidden_units', dest="hidden_units",action="store", type = int,default=1024)
parser.add_argument ('--input_units', dest="input_units",action="store", type = int,default=25088)
parser.add_argument ('--epochs',dest="epochs",action="store",  type = int,default=2)
parser.add_argument ('--gpu', dest="gpu", action="store",choices=['gpu', 'cpu'],type = str,default='gpu')
parser.add_argument ('--data_dir', dest="data_dir", action="store", default="flowers")
parser.add_argument('--category_names_mapping',  default= './cat_to_name.json')

parsed_args = parser.parse_args()
if parsed_args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'

                                                                          
  
def tst_change(test):                                     
    test_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    test_dataset = datasets.ImageFolder(test, transform=test_transform)
    return test_dataset
def val_change(val): 
    val_transform = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    val_dataset = datasets.ImageFolder(val, transform=val_transform)
    return val_dataset                  


def tr_change(train):
    train_transform = transforms.Compose([transforms.RandomRotation(45),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    train_dataset = datasets.ImageFolder(train, transform=train_transform)
    return train_dataset
                                        
def default_classif(arch,hidden_units,input_units,output_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        #freezing the parameters
        for param in model.parameters():
            param.requires_grad = False        
            classifier = nn.Sequential(OrderedDict([
             ('fc1', nn.Linear(input_units, hidden_units)),
             ('relu', nn.ReLU()),
             ('fc2', nn.Linear(hidden_units, 102)),
             ('output', nn.LogSoftmax(dim=1))]))
    else:
        arch = 'vgg19'
        model = models.vgg19(pretrained = True)
        #freezing the parameters
        for param in model.parameters():
            param.requires_grad = False        
            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(input_units, hidden_units)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(hidden_units, 102)),
                ('output', nn.LogSoftmax(dim=1))]))

    model.classifier = classifier  
    return model,arch
     
                                      
def validation(model, testloader, criterion,device):
    accuracy = 0
    test_loss = 0
    for k, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        
        accuracy += equality.type(torch.FloatTensor).mean()
        return test_loss, accuracy   
                                        
                                        
                                        
                                        
def training(model, trainloader, testloader,criterion, optimizer,device,epochs):                                        
    model.to(device)
    print_every = 1000 # Prints every 30 images out of batch of 50 images
    steps = 0
    for e in range(epochs):
        running_loss = 0
        model.train()
        for k, (inp, lbl) in enumerate(trainloader):
            steps += 1

            inputs, labels = inp.to(device), lbl.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    val_loss, accuracy = validation(model, testloader, criterion)
                print((e+1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Val Loss: {:.4f} | ".format(val_loss/len(testloader)),
                      "Val Accuracy: {:.4f}".format(accuracy/len(testloader)))

                running_loss = 0
                model.train()
    return model

def check_accuracy_on_test(model,testloader,device):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the  test images: %d %%' % (100 * correct / total))
    
def save_model(model, save_dir, train_dataset,optimizer,epochs,arch): 
    
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {

                 'classifier': model.classifier,
                 'class_to_idx': model.class_to_idx,
                  'arch': arch,
                  'epochs': epochs,
                  'optimizer_state': optimizer.state_dict(),
                 'state_dict': model.state_dict()}

    torch.save(checkpoint, save_dir + '/checkpoint.pth')
                                        
                                        
##now run all the commands
def main():   
   #load the data
    
    data_dir = parsed_args.data_dir
    train_dataset  = tr_change(data_dir+'/train')
    test_dataset = tst_change(data_dir+'/test')
    val_dataset = val_change(data_dir+'/valid')  
    
    trainloader =  torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    testloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    #capturing the amount of output classes
    with open(parsed_args.category_names_mapping, 'r') as file:
        cat_to_name = json.load(file)
    output_units_count = len(cat_to_name)
    
    
    # Load Model , depending on what user chooses vgg16 or 19
    model, arch = default_classif (parsed_args.arch,parsed_args.hidden_units,parsed_args.input_units,output_units=output_units_count)
    
    criterion = nn.NLLLoss ()
    if parsed_args.learning_rate: 
        optimizer = optim.Adam (model.classifier.parameters (), lr = parsed_args.learning_rate)
    else:
        optimizer = optim.Adam (model.classifier.parameters (), lr = 0.001)
    
   
    model.to(device);
        
    #train the model
    model_after_training = training(model, trainloader, valloader, criterion, optimizer,device,parsed_args.epochs)      
    #check results on test set
    check_accuracy_on_test(model,testloader,device)
    #save the checkpoint
    save_model(model_after_training,parsed_args.save_dir, train_dataset ,optimizer,parsed_args.epochs,parsed_args.arch)        
if __name__ == '__main__': main()