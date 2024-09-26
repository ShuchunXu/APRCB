import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models

class CNNModel(nn.Module):
    def __init__(self,num_classes):
        super(CNNModel, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        # Pooling layers
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=5)
        self.pool3 = nn.MaxPool2d(kernel_size=5, stride=5)

        self.num_classes=num_classes
                
        # Fully connected layers
        self.fc = nn.Linear(256, num_classes)  # Assuming the input image size is 100x100 and there are 6 classes

    def forward(self, x, if_feature=False):
        # Convolutional and pooling layers
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten the tensor for the fully connected layer
        x_ = x.view(x.size(0), -1)
        
        x = self.fc(x_)
        
        if if_feature:
            return x_,x
        else:
            return x

def initialize_mode(model_name,num_classes):

    if model_name=='resnet':

        model=models.resnet18(pretrained=False)

        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes) 

        input_size = 224
    
    elif model_name=='CNN':
        model=CNNModel(num_classes)
        input_size = 200

    return model,input_size

if __name__== "__main__" :
    num_classes=6
    model,input_size=initialize_mode('resnet',num_classes)
    print(model)