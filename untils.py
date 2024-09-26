import logging
import math
import os
from matplotlib import pyplot as plt
import torch
import random
import numpy as np

from collections import Counter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_aug_num(image_train):
  
    labels = image_train.targets #
    
    class_counts = Counter(labels)
    # print(class_counts)
    tmp_num=[]  #
    aug_num=[]
    for i in range(len(class_counts)):
        tmp_num.append(class_counts[i]) # （label,label's number ）

    for i in range(len(tmp_num)):
        # aug_num.append(int((max(tmp_num)/tmp_num[i])))
        aug_num.append(int(math.ceil(max(tmp_num)/tmp_num[i])))
    # print(aug_num)  #
    
    return tmp_num

def get_logger(filename, verbosity=1, name=None):  #
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def visionConfusion_Matrix(conf_matrix,image_val,result_dir,train_type): # 
    '''
    conf_matrix: save this conf_matrix
    image_val:to get all kinds of classes
    result_dir:the saving path
    '''
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(train_type+' Confusion Matrix')
    plt.colorbar()
    classes = image_val.classes  # 
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir,train_type+'_conf_matrix.png'))  #


def LineChart(conf_matrix_nature,
              conf_matrix_robustness_fgsm,
              conf_matrix_robustness_PGD,
              conf_matrix_robustness_CW,
              conf_matrix_robustness_MIM,
              conf_matrix_robustness_AA,
              results_dir,args,image_val):
    
    confusion_matrix1_nature=np.array(conf_matrix_nature)
    confusion_matrix2_fgsm=np.array(conf_matrix_robustness_fgsm) # 
    confusion_matrix2_PGD=np.array(conf_matrix_robustness_PGD) # 
    confusion_matrix2_CW=np.array(conf_matrix_robustness_CW) # 
    confusion_matrix2_MIM=np.array(conf_matrix_robustness_MIM) # 
    confusion_matrix2_AA=np.array(conf_matrix_robustness_AA) # 

    #
    class_accuracies1_nature = []
    class_accuracies2_FGSM = []
    class_accuracies2_PGD = []
    class_accuracies2_CW = []
    class_accuracies2_MIM = []
    class_accuracies2_AA = []

    for i in range(confusion_matrix1_nature.shape[0]):
        
        # natural
        true_positives1_nature = confusion_matrix1_nature[i, i]
        total_samples1_nature = np.sum(confusion_matrix1_nature[i, :])
        accuracy1_nature = true_positives1_nature / total_samples1_nature
        class_accuracies1_nature.append(accuracy1_nature)  # 

        # fgsm
        true_positives2_fgsm = confusion_matrix2_fgsm[i, i]
        total_samples2_fgsm = np.sum(confusion_matrix2_fgsm[i, :])
        accuracy2_fgsm = true_positives2_fgsm / total_samples2_fgsm
        class_accuracies2_FGSM.append(accuracy2_fgsm) 

        # PGD
        true_positives2_PGD = confusion_matrix2_PGD[i, i]
        total_samples2_PGD = np.sum(confusion_matrix2_PGD[i, :])
        accuracy2_PGD = true_positives2_PGD / total_samples2_PGD
        class_accuracies2_PGD.append(accuracy2_PGD) 

        # CW
        true_positives2_CW = confusion_matrix2_CW[i, i]
        total_samples2_CW = np.sum(confusion_matrix2_CW[i, :])
        accuracy2_CW = true_positives2_CW / total_samples2_CW
        class_accuracies2_CW.append(accuracy2_CW) # 

        # MIM
        true_positives2_MIM = confusion_matrix2_MIM[i, i]
        total_samples2_MIM = np.sum(confusion_matrix2_MIM[i, :])
        accuracy2_MIM = true_positives2_MIM / total_samples2_MIM
        class_accuracies2_MIM.append(accuracy2_MIM) # 

        # AA
        true_positives2_AA = confusion_matrix2_AA[i, i]
        total_samples2_AA = np.sum(confusion_matrix2_AA[i, :])
        accuracy2_AA = true_positives2_AA / total_samples2_AA
        class_accuracies2_AA.append(accuracy2_AA) #


    with open(os.path.join(results_dir,args.train_type+'_exp.log'),'a') as f:
       

        
        print('nature acc:',file=f,flush=True) # 
        for i, acc in enumerate(class_accuracies1_nature):  
            print(f"Class {i+1} Accuracy: {acc}",file=f,flush=True)

        print(f"",file=f,flush=True)

        print('FGSM\'s adversarial acc:',file=f,flush=True) 
        for i, acc in enumerate(class_accuracies2_FGSM):
            print(f"Class {i+1} Accuracy: {acc}",file=f,flush=True)

        print(f"",file=f,flush=True)

        print('PGD\'s adversarial acc:',file=f,flush=True) #
        for i, acc in enumerate(class_accuracies2_PGD):
            print(f"Class {i+1} Accuracy: {acc}",file=f,flush=True)

        print(f"",file=f,flush=True)

        print('CW\'s adversarial acc:',file=f,flush=True) 
        for i, acc in enumerate(class_accuracies2_CW):
            print(f"Class {i+1} Accuracy: {acc}",file=f,flush=True)

        print(f"",file=f,flush=True)

        print('MIM\'s adversarial acc:',file=f,flush=True) #
        for i, acc in enumerate(class_accuracies2_MIM):
            print(f"Class {i+1} Accuracy: {acc}",file=f,flush=True)

        print(f"",file=f,flush=True)

        print('AA\'s adversarial acc:',file=f,flush=True) #
        for i, acc in enumerate(class_accuracies2_AA):
            print(f"Class {i+1} Accuracy: {acc}",file=f,flush=True)


  
    plt.clf()#
    classes = image_val.classes  #
    tick_marks = np.arange(len(classes))
    class_names=[x for x in range(0,args.num_classes)]


    plt.plot(class_names, class_accuracies1_nature, marker='o', label='nature acc')
    plt.plot(class_names, class_accuracies2_FGSM, marker='o', label='robustness acc of FGSM') # 
    plt.plot(class_names, class_accuracies2_PGD, marker='o', label='robustness acc of PGD') #
    plt.plot(class_names, class_accuracies2_CW, marker='o', label='robustness acc of CW') 
    plt.plot(class_names, class_accuracies2_MIM, marker='o', label='robustness acc of MIM') #
    plt.plot(class_names, class_accuracies2_AA, marker='o', label='robustness acc of AA') #



    plt.title('Class Accuracies')
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    # plt.xticks(tick_marks,classes)
    plt.xticks()
    plt.ylim(0, 1)  
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,args.train_type+'.png'))  