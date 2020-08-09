# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


## The image classifier to recognize different species of flowers. Dataset contains 102 flower categories.

In Image Classifier Project.ipynb VGG16(a convolutional neural network model proposed by K. Simonyan and A. Zisserman from the University of Oxford) from torchvision.models pretrained models was used. 
The model was loaded as a pre-trained model, and based on this model I defined a classifier of my own (feed=forward). 
I used Loss and accuracy as my metrics to define model's success

## Code walkthrough
For the 1st part, the code is provided in the Image Classifier Project.ipynb notebook file.

For the 2nd part, the code is in the train.py and predict.py file. In addition to these, you'll also need the files cat_to_name.json for predict.py to run the prediction using  a mapping of flower categories to real names

## Command line instructions for  train.py and predict.py

In oder to successfully run train.py and predict.py the most basic level of arguments you would need to run is :
```
python ./ImageClassifier/train.py './ImageClassifier/flowers'
```
for train.py

and
```
python  ./ImageClassifier/predict.py --image_path './ImageClassifier/flowers/valid/10/image_07094.jpg' --checkpoint './ImageClassifier/checkpoint.pth' --category_names './ImageClassifier/cat_to_name.json'
```
For predict.py
