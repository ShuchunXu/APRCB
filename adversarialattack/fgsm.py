import torch
import torch.nn as nn
from torch.autograd import Variable

def fgsm_attack_eval(model,inputs,labels):
    epsilon=0.031
    X_fgsm = Variable(inputs.data, requires_grad=True)
    with torch.enable_grad(): 
        loss = nn.CrossEntropyLoss()(model(X_fgsm), labels)
    loss.backward()
    eta = epsilon * X_fgsm.grad.data.sign()  
    X_fgsm = Variable(inputs.data + eta, requires_grad=True)  
    X_fgsm = Variable(torch.clamp(X_fgsm, 0, 1.0), requires_grad=True)

    return X_fgsm
