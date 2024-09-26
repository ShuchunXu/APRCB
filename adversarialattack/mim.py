import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

def MimAttack(model_cnn, x_natural, y_batch,
               nb_it=20, eta=0.01, epsilon= 8. / 255.,rand_start_mode='uniform',
               distance='l_inf',rand_start_step=1): # eta=2 / 255 
    

    model_cnn.eval()  
    decay_factor=1.0

    if rand_start_mode == 'gaussian':
        x_adv = x_natural.detach() + rand_start_step * 0.001 * torch.randn(x_natural.shape).cuda().detach()
    elif rand_start_mode == 'uniform': 
        x_adv = x_natural.detach() + rand_start_step * epsilon * torch.rand(x_natural.shape).cuda().detach() 
    else:
        raise NameError
    previous_grad = torch.zeros_like(x_adv.data) 


    if distance == 'l_inf':
        for _ in range(nb_it):  
            x_adv.requires_grad_()

            with torch.enable_grad():
                loss_ce = nn.CrossEntropyLoss()(model_cnn(x_adv), y_batch) 
            loss_ce.backward()
            grad = x_adv.grad.data / torch.mean(torch.abs(x_adv.grad.data), [1,2,3], keepdim=True)
            previous_grad = decay_factor * previous_grad + grad



            x_adv = x_adv.detach() + eta * previous_grad.sign()
            x_adv = torch.min( torch.max(x_adv, x_natural - epsilon) , x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0) 
    else:
        raise NotImplementedError

    X_att = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    return X_att.detach()


def MIM_attack_eval( model,inputs,labels):
    
    x_MimAttack=MimAttack(model,inputs, labels)
    
    return x_MimAttack
