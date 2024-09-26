import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from adversarialattack.pgd import *

def get_effectNum(sample_pre_class,beta):
    effective_num = 1.0 - np.power(beta, sample_pre_class) #
    return effective_num 

def natrualTest(model,val_loader,device,loss_func):  #
    model.eval()
    running_corrects=0
    running_loss=0
    for inputs,labels in val_loader: 
        inputs,labels=inputs.to(device),labels.to(device)

        # forward--
        with torch.no_grad():
            outputs=model(inputs)
        loss=loss_func(outputs,labels)

        # statistic
        _,preds=torch.max(outputs,1)

        running_corrects += torch.sum(preds==labels.data) #
        running_loss += loss.item() #

    epoch_loss=running_loss*1.0/len(val_loader)
    epoch_acc=running_corrects*1.0/len(val_loader.dataset)
    
    return epoch_acc,epoch_loss  #

def advTest(model,val_loader,device,loss_func):# --pgd
    model.eval()
    running_corrects=0
    running_loss=0
    for inputs,labels in val_loader: 
        inputs,labels=inputs.to(device),labels.to(device)

       
        pgd_inputs=pgd_attack_test(model,inputs,labels,loss_func,nb_it=10)  # used pgd-20 to generate adversarial example
        # forward-
        with torch.no_grad():
            outputs=model(pgd_inputs)
        loss=loss_func(outputs,labels) # 

        # statistic
        _,preds=torch.max(outputs,1)

        running_corrects+=torch.sum(preds==labels.data) # 
        running_loss+=loss.item() 

    epoch_loss=running_loss*1.0/len(val_loader)
    epoch_acc=running_corrects*1.0/len(val_loader.dataset)
    
    return epoch_acc,epoch_loss  # 

def naturalTrain(model,args,train_loader,val_loader,device,optimizer,loss_func,scheduler,writter,logger):# 

    for epoch in range(1,args.num_epochs+1): 
        print('{}/{} Epoch'.format(epoch,args.num_epochs))
        print('-'*10)
        #
        for param_group in optimizer.param_groups:
            print("当前学习率: ", param_group['lr'])

        model.train() #
        running_corrects=0
        running_loss=0
        for inputs,labels in train_loader: #
            inputs,labels =inputs.to(device),labels.to(device) 
            optimizer.zero_grad()

            # forward
            outputs=model(inputs)
            loss=loss_func(outputs,labels)

            # backforward
            loss.backward()
            optimizer.step()

            # statistics
            _,preds=torch.max(outputs,1)
            
            running_corrects+=torch.sum(preds==labels.data) # 
            running_loss+=loss.item()  

        epoch_loss=running_loss*1.0/len(train_loader)
        epoch_acc=running_corrects*1.0/len(train_loader.dataset)
        print('Trainning: Loss:{:.4f} Acc:{:.4f}'.format(epoch_loss,epoch_acc))

        writter.add_scalars('loss',  {'train': epoch_loss}, epoch)
        writter.add_scalars('acc', {'train': epoch_acc}, epoch) 
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.num_epochs, epoch_loss, epoch_acc))
        scheduler.step()

        epoch_acc,epoch_loss=natrualTest(model,val_loader,device,loss_func)  

        print('Test: Loss:{:.4f} Acc:{:.4f}'.format(epoch_loss,epoch_acc))

        writter.add_scalars('loss',  {'test': epoch_loss}, epoch)
        writter.add_scalars('acc', {'test': epoch_acc}, epoch) 
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.num_epochs, epoch_loss, epoch_acc))


    return model

def advTrain(model,args,train_loader,val_loader,device,optimizer,loss_func,scheduler,writter,logger,samples_per_class):

    kl = nn.KLDivLoss( size_average='none' ).cuda()   
    tail_class = [ i for i in range(len(samples_per_class)//3 * 2, len(samples_per_class)) ]

  
    beta = 0.999 # 0
    E = get_effectNum(samples_per_class,beta) 
    weight = (1.0 - beta) / np.array(E)
    weight = weight / np.sum(weight) * int(args.num_classes)
    weight=torch.from_numpy( weight ) 
    
  
    spc = torch.tensor(samples_per_class)  
    weights = torch.sqrt(1. / (spc / spc.sum())) 
    # weights=weight

    for epoch in range(1,args.num_epochs+1):
        print('{}/{} Epoch'.format(epoch,args.num_epochs))
        print('-'*10)
     
        for param_group in optimizer.param_groups:
            print('当前学习率：',param_group['lr'])
        
        model.train() 
        running_corrects=0
        running_loss=0
        for inputs,labels in train_loader:
            inputs,labels=inputs.to(device),labels.to(device)
            optimizer.zero_grad() 

           
            if 'our' in args.train_type:# --
                pgd_inputs=pgd_attack_train_our(model,inputs,labels,loss_func,samples_per_class) # --
            else: # standard training 
                pgd_inputs=pgd_attack_train(model,inputs,labels,loss_func)  #

            # forward--
            if 'our' in args.train_type:
                f_adv, outputs = model(pgd_inputs, True)    
                TAIL = 0.0 
                counter = 0.0
                for bi in range( labels.size(0) ):  
                    if labels[bi].item() in tail_class: 
                        idt = torch.tensor( [-1. if labels[bi].item()==labels[bj].item()  else 1. for bj in range(labels.size(0))] ).cuda()
                        W = torch.tensor( [weights[labels[bi].item()] + weights[labels[bj].item()] for bj in range(labels.size(0))] ).cuda()
                        TAIL += kl( F.log_softmax(f_adv, 1), F.softmax( f_adv[bi, :].clone().detach().view(1, -1).tile(labels.size(0), ).view(labels.size(0), -1), 1)) * idt * W
                        counter += 1
                TAIL = TAIL.mean() / counter if counter > 0. else 0.0


                weight=weight.to(outputs.dtype).to(device)
                weights=weight
                print( 'weight===:', weight ) 
                loss =  F.cross_entropy(input=outputs, target=labels,weight=weight) + TAIL  

            else: # standard training
                outputs=model(pgd_inputs) 
                loss=loss_func(outputs,labels)

            # backforward 
            loss.backward() 
            optimizer.step()

            # statistic
            _,preds=torch.max(outputs,1) 

            running_corrects+=torch.sum(preds==labels.data)
            running_loss+=loss.item()
        
        epoch_loss = running_loss*1.0/len(train_loader)
        epoch_acc  = running_corrects*1.0/len(train_loader.dataset)
        print('Trainning: Loss:{:.4f} Acc:{:.4f}'.format(epoch_loss,epoch_acc))

        writter.add_scalars('loss',  {'train': epoch_loss}, epoch)
        writter.add_scalars('acc', {'train': epoch_acc}, epoch) 
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t RobustAcc={:.3f}'.format(epoch , args.num_epochs, epoch_loss, epoch_acc))
        scheduler.step()

        epoch_acc,epoch_loss=advTest( model,val_loader,device,loss_func ) 
             
        print('Test: Loss:{:.4f} RobustAcc:{:.4f}'.format(epoch_loss,epoch_acc))

        writter.add_scalars('loss',  {'test': epoch_loss}, epoch)
        writter.add_scalars('acc', {'test': epoch_acc}, epoch) 
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t RobustAcc={:.3f}'.format(epoch , args.num_epochs, epoch_loss, epoch_acc))

    return model



        