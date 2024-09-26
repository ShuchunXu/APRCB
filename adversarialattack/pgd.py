import numpy as np
import torch

from torch.autograd import Variable


def pgd_attack_test(model_cnn,x_natural,y_batch,loss_model,nb_it=20 ):

    rand_start_mode='uniform'
    rand_start_step=1
    distance='l_inf'

    eta=  0.01
    epsilon=8. / 255.  

    model_cnn.eval()

    if rand_start_mode == 'gaussian':
        x_adv = x_natural.detach() + rand_start_step * 0.001 * torch.randn(x_natural.shape).cuda().detach()
    elif rand_start_mode == 'uniform': 
        x_adv = x_natural.detach() + rand_start_step * 8. / 255. * torch.rand(x_natural.shape).cuda().detach() 
        # 初始化生成对抗样本
    else:
        raise NameError
    if distance == 'l_inf' :
        for _ in range(nb_it):  
            x_adv.requires_grad_() 

            with torch.enable_grad():
                loss_ce = loss_model(model_cnn(x_adv), y_batch) 


            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + eta * torch.sign( grad.detach() ) 
            x_adv = torch.min( torch.max(x_adv, x_natural - epsilon) , x_natural + epsilon) 
            x_adv = torch.clamp(x_adv, 0.0, 1.0)  
    else:
        raise NotImplementedError

    X_att = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    model_cnn.train()
    return X_att.detach() 

def pgd_attack_train(model_cnn,x_natural,y_batch,loss_model):
    # 常参数
    rand_start_mode='uniform'
    rand_start_step=1
    distance='l_inf'
    nb_it=10 
    eta=  0.01
    epsilon=8. / 255.  

    model_cnn.eval()  

    if rand_start_mode == 'gaussian':
        x_adv = x_natural.detach() + rand_start_step * 0.001 * torch.randn(x_natural.shape).cuda().detach()
    elif rand_start_mode == 'uniform':

        x_adv = x_natural.detach() + rand_start_step * 8. / 255. * torch.rand(x_natural.shape).cuda().detach() 

    else:
        raise NameError
    
    if distance == 'l_inf' :
        for _ in range(nb_it):  
            x_adv.requires_grad_() 
            with torch.enable_grad():
                loss_ce = loss_model(model_cnn(x_adv), y_batch) 


            grad = torch.autograd.grad(loss_ce, [x_adv])[0] 
            x_adv = x_adv.detach() + eta * torch.sign( grad.detach() )
            x_adv = torch.min( torch.max(x_adv, x_natural - epsilon) , x_natural + epsilon) 
            x_adv = torch.clamp(x_adv, 0.0, 1.0) 
    else:
        raise NotImplementedError

    X_att = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    model_cnn.train()
    return X_att.detach()



def simplified_model_func(x, a, b, c, d):
    return a * np.log(x) + b * np.exp(-c * x) + d

def get_epsilon(samples_per_class):


    a_simplified = -0.016857350721965245
    b_simplified = 0.0007347335137892177
    c_simplified = 0.02490244955936155
    d_simplified = 0.12376449284587332
    # 定义数据
    number = np.array(samples_per_class)


    epsilon_calculated = simplified_model_func(number, a_simplified, b_simplified, c_simplified, d_simplified)
    epsilon_list = epsilon_calculated.tolist()
    return epsilon_list

def pgd_attack_train_our(model_cnn,x_natural,y_batch,loss_model,samples_per_class):
    # 常参数
    rand_start_mode='uniform'
    rand_start_step=1
    distance='l_inf'
    nb_it=10  
    eta=0.01
    epsilon=get_epsilon(samples_per_class)

    model_cnn.eval() 

    if rand_start_mode == 'gaussian':
        x_adv = x_natural.detach() + rand_start_step * 0.001 * torch.randn(x_natural.shape).cuda().detach()
    elif rand_start_mode == 'uniform':
        x_adv = x_natural.detach() + rand_start_step * (8. / 255.) * torch.rand(x_natural.shape).cuda().detach() 

    else:
        raise NameError
    
    if distance == 'l_inf' :
        for _ in range(nb_it):  
            x_adv.requires_grad_() 
            with torch.enable_grad():
                loss_ce = loss_model(model_cnn(x_adv), y_batch) 


            grad = torch.autograd.grad(loss_ce, [x_adv])[0]
            x_adv = x_adv.detach() + eta * torch.sign( grad.detach() ) 

            for i in range(0,x_adv.size()[0]):
                x_adv[i]=torch.min( torch.max(x_adv[i], x_natural[i] - epsilon[ y_batch[i].item() ] ) , x_natural[i] + epsilon[ y_batch[i].item() ] ) 
            x_adv = torch.clamp(x_adv, 0.0, 1.0) 
    else:
        raise NotImplementedError

    X_att = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    model_cnn.train() 
    return X_att.detach()