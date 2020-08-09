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
vgg16 = models.vgg16(pretrained=True)
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
parsed_args = parser.parse_args()
if type(parsed_args.learning_rate) == type(None):
        learning_rate = 0.001
else: learning_rate = parsed_args.learning_rate

                                                                          
  
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
                                        
def default_classif():
    classifier = nn.Sequential(OrderedDict([
     ('fc1', nn.Linear(25088, 1024)),
     ('relu', nn.ReLU()),
     ('fc2', nn.Linear(1024, 102)),
     ('output', nn.LogSoftmax(dim=1))]))
    return classifier
     
                                      
def validation(model, testloader, criterion):
    accuracy = 0
    test_loss = 0
    for k, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        
        accuracy += equality.type(torch.FloatTensor).mean()
        return test_loss, accuracy   
                                        
                                        
                                        
                                        
def training(model, trainloader, testloader,criterion, optimizer):                                        
    model.to('cuda')
    epochs = 2
    print_every = 1000 # Prints every 30 images out of batch of 50 images
    steps = 0
    for e in range(epochs):
        running_loss = 0
        model.train()
        for k, (inp, lbl) in enumerate(trainloader):
            steps += 1

            inputs, labels = inp.to('cuda'), lbl.to('cuda')

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
                    val_loss, accuracy = validation(model, valloader, criterion)
                print((e+1, epochs),
                      "Training Loss: {:.4f} | ".format(running_loss/print_every),
                      "Val Loss: {:.4f} | ".format(val_loss/len(testloader)),
                      "Val Accuracy: {:.4f}".format(accuracy/len(testloader)))

                running_loss = 0
                model.train()
    return model

def check_accuracy_on_test(model,testloader):    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the  test images: %d %%' % (100 * correct / total))
    
def save_model(model, save_dir, train_dataset,optimizer): 
    epochs = 2
    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {

                 'classifier': model.classifier,
                 'class_to_idx': model.class_to_idx,
                  'arch': 'vgg16',
                  'epochs': epochs,
                  'optimizer_state': optimizer.state_dict(),
                 'state_dict': model.state_dict()}

    torch.save(checkpoint, 'my_checkpoint.pth')
                                        
                                        
##now run all the commands
def main():   
   #load the data
    
    data_dir = 'flowers'
    train_dataset  = tr_change(data_dir+'/train')
    test_dataset = tst_change(data_dir+'/test')
    val_dataset = val_change(data_dir+'/valid')  
    
    trainloader =  torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    testloader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    # Load Model
    model = models.vgg16(pretrained=True)
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False 
    model.classifier = default_classif()
    model.to('cuda');
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    print_every = 1000
    steps = 0
    
    #train the model
    model_after_training = training(model, trainloader, valloader, criterion, optimizer)      
    #check results on test set
    check_accuracy_on_test(model,testloader)
    #save the checkpoint
    save_model(model_after_training, parsed_args.save_dir, train_dataset ,optimizer)        
if __name__ == '__main__': main()