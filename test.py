from collections import Counter
import os
import torch
import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from adversarialattack.cw import cw_attack_eval

from adversarialattack.fgsm import fgsm_attack_eval
from adversarialattack.mim import MIM_attack_eval
from adversarialattack.pgd import pgd_attack_test
from autoattack import AutoAttack
from untils import LineChart, visionConfusion_Matrix  

def auto_attack(model):

    model.eval()
    test_epsilon=0.031
    # attacks_to_run = ['apgd-ce', 'apgd-t']#, 'apgd-dlr', 'fab-t']
    attacks_to_run = ['apgd-ce', 'apgd-dlr']#, 'apgd-dlr', 'fab-t']
    adversary = AutoAttack(model, norm='Linf', eps=test_epsilon, version='standard',verbose=False)
    adversary.attacks_to_run = attacks_to_run


    return adversary

def CalculateRobustAccuracy_fgsm(model,val_loader,args,conf_matrix_robustness_fgsm,device):
    corrects_att=0
    model.eval()
    for inputs,labels in val_loader:
        inputs,labels = inputs.to(device),labels.to(device)
        
        inputs_att=fgsm_attack_eval(model,inputs,labels)

        # forward
        with torch.no_grad():
            outputs=model(inputs_att) 

        # statistics
        _,preds_att=torch.max(outputs,1)
        corrects_att+=torch.sum(preds_att==labels.data)
        preds_att_list = preds_att.tolist()
        labels_list=labels.tolist()
        
        conf_matrix_robustness_fgsm += confusion_matrix(labels_list,preds_att_list,labels=[x for x in range(0,args.num_classes)])

    acc_fgsm=(corrects_att*1.0) / len(val_loader.dataset)

    return acc_fgsm,conf_matrix_robustness_fgsm

def CalculateRobustAccuracy_PGD(model,val_loader,args,conf_matrix_robustness_PGD,device):
    corrects_att=0
    model.eval()
    for inputs,labels in val_loader:
        inputs,labels=inputs.to(device),labels.to(device)

        # inputs_att=pgd_attack_eval( model,device,inputs,labels,nn.CrossEntropyLoss(),nb_it=20 ) 
        inputs_att=pgd_attack_test(model,inputs,labels,nn.CrossEntropyLoss(),nb_it=20)
        # inputs_att: 为迭代 20 轮之后生成对抗样本
        # 迭代 20 轮
        # forward
        with torch.no_grad():
            outputs_att=model(inputs_att) 

        _,preds_att=torch.max(outputs_att,1) 
        corrects_att += torch.sum(preds_att==labels.data) 
        preds_att_list = preds_att.tolist()
        labels_list=labels.tolist()

        conf_matrix_robustness_PGD += confusion_matrix(labels_list,preds_att_list,labels=[x for x in range(0,args.num_classes)])

    acc_pgd=(corrects_att*1.0) / len(val_loader.dataset)

    return acc_pgd,conf_matrix_robustness_PGD

def CalculateRobustAccuracy_CW(model,val_loader,args,conf_matrix_robustness_CW,device):
    corrects_att=0
    model.eval() 
    for inputs,labels in val_loader:
        inputs,labels=inputs.to(device),labels.to(device)

        inputs_att=cw_attack_eval( model,inputs,labels )

         # forward
        with torch.no_grad():
            outputs_att=model(inputs_att)

        _,preds_att=torch.max(outputs_att,1)
        corrects_att += torch.sum(preds_att==labels.data) 
        preds_att_list = preds_att.tolist()
        labels_list=labels.tolist()

        conf_matrix_robustness_CW += confusion_matrix(labels_list,preds_att_list,labels=[x for x in range(0,args.num_classes)])

    acc_cw=(corrects_att*1.0) / len(val_loader.dataset)

    return acc_cw,conf_matrix_robustness_CW

def CalculateRobustAccuracy_MiM(model,val_loader,args,conf_matrix_robustness_MIM,device):
    corrects_att=0
    model.eval()
    for inputs,labels in val_loader:
        inputs,labels=inputs.to(device),labels.to(device)

        inputs_att=MIM_attack_eval( model,inputs,labels)

         # forward
        with torch.no_grad():
            outputs_att=model(inputs_att)

        _,preds_att=torch.max(outputs_att,1)
        corrects_att += torch.sum(preds_att==labels.data) 
        preds_att_list = preds_att.tolist()
        labels_list=labels.tolist()

        conf_matrix_robustness_MIM += confusion_matrix(labels_list,preds_att_list,labels=[x for x in range(0,args.num_classes)])

    acc_mim=(corrects_att*1.0) / len(val_loader.dataset)

    return acc_mim,conf_matrix_robustness_MIM

def CalculateRobustAccuracy_AA(model,val_loader,args,conf_matrix_robustness_AA,device):
    corrects_att=0
    model.eval()
    adversary=auto_attack(model)
    for inputs,labels in val_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        
        # inputs_att = adversary.run_standard_evaluation_individual(inputs, labels, bs=len(inputs))
        inputs_att = adversary.run_standard_evaluation(inputs, labels, bs=len(inputs))
        # pdb.set_trace()
         # forward
        with torch.no_grad():
            outputs_att=model(inputs_att)

        _,preds_att=torch.max(outputs_att,1)
        corrects_att += torch.sum(preds_att==labels.data) 
        preds_att_list = preds_att.tolist()
        labels_list=labels.tolist()

        conf_matrix_robustness_AA += confusion_matrix(labels_list,preds_att_list,labels=[x for x in range(0,args.num_classes)])

    acc_aa=(corrects_att*1.0) / len(val_loader.dataset)

    return acc_aa,conf_matrix_robustness_AA

def FinallyTest(model,args,val_loader,image_val,image_train,results_dir,data_dir,device):

    model.eval() 
    corr=0 

    conf_matrix_nature          = np.zeros((args.num_classes, args.num_classes), dtype=int ) 
    conf_matrix_robustness_fgsm = np.zeros((args.num_classes, args.num_classes), dtype=int ) 
    conf_matrix_robustness_PGD  = np.zeros((args.num_classes, args.num_classes), dtype=int ) 
    conf_matrix_robustness_CW   = np.zeros((args.num_classes, args.num_classes), dtype=int )
    conf_matrix_robustness_MIM  = np.zeros((args.num_classes, args.num_classes), dtype=int ) 
    conf_matrix_robustness_AA   = np.zeros((args.num_classes, args.num_classes), dtype=int )



    for inputs,labels in val_loader:
        inputs,labels=inputs.to(device),labels.to(device)

        # forward
        with torch.no_grad():
            outputs=model(inputs)

        _,preds=torch.max(outputs,1)
        corr+=torch.sum(preds==labels.data)
        preds_list=preds.tolist()
        labels_list=labels.tolist()
        conf_matrix_nature += confusion_matrix( labels_list,preds_list,labels=[x for x in range(0,args.num_classes)] ) # 
    
    natural_acc=(corr*1.0) / len(val_loader.dataset) # 计算自然acc

   
    acc_fgsm,conf_matrix_robustness_fgsm=CalculateRobustAccuracy_fgsm(model,val_loader,args,conf_matrix_robustness_fgsm,device)

   
    acc_pgd,conf_matrix_robustness_PGD=CalculateRobustAccuracy_PGD(model,val_loader,args,conf_matrix_robustness_PGD,device)

   
    acc_cw,conf_matrix_robustness_CW=CalculateRobustAccuracy_CW(model,val_loader,args,conf_matrix_robustness_CW,device)

   
    acc_mim,conf_matrix_robustness_MIM=CalculateRobustAccuracy_MiM(model,val_loader,args,conf_matrix_robustness_MIM,device)

   
    acc_aa,conf_matrix_robustness_AA=CalculateRobustAccuracy_AA(model,val_loader,args,conf_matrix_robustness_AA,device)

    # 
    print('nature acc:{0:.4f}'.format(natural_acc)) 
    print('conf_matrix_nature:')
    print(conf_matrix_nature)
    print('')

    print('Robustness acc(FGSM):{:.4f}'.format(acc_fgsm)) #
    print('conf_matrix_robustness_FGSM:')
    print(conf_matrix_robustness_fgsm )
    print('')

    print('Robustness acc(PGD):{:.4f}'.format(acc_pgd)) #
    print('conf_matrix_robustness_PGD:')
    print(conf_matrix_robustness_PGD )
    print('')

    print('Robustness acc(CW):{:.4f}'.format(acc_cw))
    print('conf_matrix_robustness_CW:')
    print(conf_matrix_robustness_CW )
    print('')

    print('Robustness acc(MIM):{:.4f}'.format(acc_mim)) # 
    print('conf_matrix_robustness_MIM:')
    print(conf_matrix_robustness_MIM )
    print('')

    print('Robustness acc(AA):{:.4f}'.format(acc_aa)) # 
    print('conf_matrix_robustness_AA:')
    print(conf_matrix_robustness_AA )
    print('')

   
    visionConfusion_Matrix(conf_matrix_nature,image_val,results_dir,'clear')  #
    visionConfusion_Matrix(conf_matrix_robustness_fgsm,image_val,results_dir,'FGSM\'s robustness') #
    visionConfusion_Matrix(conf_matrix_robustness_PGD,image_val,results_dir,'PGD\'s robustness') 
    visionConfusion_Matrix(conf_matrix_robustness_CW,image_val,results_dir,'C&W\'s robustness') 
    visionConfusion_Matrix(conf_matrix_robustness_MIM,image_val,results_dir,'MIM\'s robustness') 
    visionConfusion_Matrix(conf_matrix_robustness_AA,image_val,results_dir,'AA\'s robustness')
    
  
    LineChart(conf_matrix_nature,
              conf_matrix_robustness_fgsm,
              conf_matrix_robustness_PGD,
              conf_matrix_robustness_CW,
              conf_matrix_robustness_MIM,
              conf_matrix_robustness_AA,
              results_dir,args,image_val)


    print(image_train.class_to_idx)
        
    labels = image_train.targets

    class_counts = Counter(labels)

   
    list_num=[]
    print("Number of images per class:")
    for class_idx, count in class_counts.items():
        list_num.append(count)
        class_name = image_train.classes[class_idx]
        print(f"{class_name}: {count}")

    with open(os.path.join(results_dir,args.train_type+'_exp.log'),'a') as f:
        print("共训练{0}轮".format(args.num_epochs),file=f,flush=True)

        print("clear acc:{:.4f}".format(natural_acc),file=f,flush=True) # 
        print('conf_matrix_nature:',file=f,flush=True)
        print(conf_matrix_nature,file=f,flush=True)

        print(f"",file=f,flush=True)

        print("robustness acc(FGSM):{:.4f}".format(acc_fgsm),file=f,flush=True) # 
        print('conf_matrix_robustness_FGSM:',file=f,flush=True) 
        print(conf_matrix_robustness_fgsm,file=f,flush=True)

        print(f"",file=f,flush=True)

        print("robustness acc(PGD):{:.4f}".format(acc_pgd),file=f,flush=True)
        print('conf_matrix_robustness_PGD:',file=f,flush=True)  #
        print(conf_matrix_robustness_PGD,file=f,flush=True)

        print(f"",file=f,flush=True)

        print("robustness acc(CW):{:.4f}".format(acc_cw),file=f,flush=True)
        print('conf_matrix_robustness_CW:',file=f,flush=True)        
        print(conf_matrix_robustness_CW,file=f,flush=True)

        print(f"",file=f,flush=True)

        print("robustness acc(MIM):{:.4f}".format(acc_mim),file=f,flush=True)
        print('conf_matrix_robustness_MIM:',file=f,flush=True)         
        print(conf_matrix_robustness_MIM,file=f,flush=True)

        print(f"",file=f,flush=True)

        print("robustness acc(AA):{:.4f}".format(acc_aa),file=f,flush=True)
        print('conf_matrix_robustness_AA:',file=f,flush=True)      
        print(conf_matrix_robustness_AA,file=f,flush=True)

        print(f"",file=f,flush=True)

        print("Number of images per class:",file=f,flush=True)
        for class_idx, count in class_counts.items():
            class_name = image_train.classes[class_idx]
            print(f"{class_name}: {count}",file=f,flush=True)
        print('data_dir:',data_dir,file=f,flush=True)
        print('result_dir:',results_dir,file=f,flush=True)

        print(f"",file=f,flush=True)

        print(args,file=f,flush=True)
        print(torch.cuda.get_device_name(),file=f,flush=True)