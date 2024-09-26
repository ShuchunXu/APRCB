import argparse
import os
import pdb
from scipy import optimize
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from untils import *
from Models import *
from getData import *
from train import *
from test import *

from torch.utils.tensorboard import SummaryWriter
# 0.Device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
set_seed(42)


parser=argparse.ArgumentParser()

## training
parser.add_argument('--lr',type=float,default=0.001,help='learn rate')
parser.add_argument('--num_epochs', type=int, default=0, help='max epoch')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--train_type', type=str, default='natural', help='the type of training')

## data
parser.add_argument('--batchSize', type=int, default=32, help='batch size')
parser.add_argument('--num_classes', type=int, default=6, help='number of classes')
parser.add_argument('--imbalance', type=float, default=1.0, help='imbalance factor')

## model
parser.add_argument('--model_name', type=str, default='CNN', help='number of classes')

args = parser.parse_args()
print(args)


data_dir = '/mandapeng03/Experiment/dataset/AdversarialTrainDataset/NEU-CLS_'+str(args.imbalance)
result_path=os.path.join( './result_'+str(args.num_epochs),str(args.imbalance),args.train_type) 
if not os.path.exists(result_path):
    os.makedirs(result_path)


model,input_size=initialize_mode(args.model_name,args.num_classes)
model=model.to(device)


train_loader,val_loader,test_loader,image_train,image_val,image_test=getdatas(input_size,data_dir,args.batchSize,data_name='NEU-CLS')
samples_per_class=get_aug_num(image_train) 

print(train_loader)


loss_func = nn.CrossEntropyLoss() 
optimizer = optim.Adam( [{'params': model.parameters(), 'lr': args.lr}], betas=(0.5, 0.999) ) 

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, eta_min=0)


writter = SummaryWriter(result_path+'/tensorboard') 
logger = get_logger(  os.path.join(result_path,args.train_type+'_exp.log') )  


if 'natural' in args.train_type:
    model=naturalTrain(model,args,train_loader,val_loader,device,optimizer,loss_func,scheduler,writter,logger)
elif 'adversarial' in args.train_type: 
    model=advTrain(model,args,train_loader,val_loader,device,optimizer,loss_func,scheduler,writter,logger,samples_per_class)

torch.save(model,os.path.join(result_path,'{}.pt'.format(args.model_name)))


FinallyTest(model,args,test_loader,image_test,image_train,result_path,data_dir,device)

print('done')
