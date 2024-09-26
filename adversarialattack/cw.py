import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable



def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor

class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes  
        self.margin = margin 
        self.reduce = reduce  
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes,targets.device) 


        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max((1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss
    
def CW_attack(model_cnn, x_natural, y_batch,
               nb_it=20, eta=0.01, epsilon= 8. / 255.,rand_start_mode='uniform',
               distance='l_inf',rand_start_step=1): 
    

    model_cnn.eval() 

    if rand_start_mode == 'gaussian':
        x_adv = x_natural.detach() + rand_start_step * 0.001 * torch.randn(x_natural.shape).cuda().detach()
    elif rand_start_mode == 'uniform':
        x_adv = x_natural.detach() + rand_start_step * epsilon * torch.rand(x_natural.shape).cuda().detach() 
    else:
        raise NameError
    
    if distance == 'l_inf':
        for _ in range(nb_it):  
            x_adv.requires_grad_()

            with torch.enable_grad():
                loss = CWLoss(6)(model_cnn(x_adv), y_batch) 

            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + eta * torch.sign(grad.detach())
            x_adv = torch.min( torch.max(x_adv, x_natural - epsilon) , x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0) 
    else:
        raise NotImplementedError

    X_att = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    
    return X_att.detach() 



def cw_attack_eval(model,images,labels):
    X_cw=CW_attack(model,images,labels)
    return X_cw
