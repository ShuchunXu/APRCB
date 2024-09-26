import os
import torch

from torchvision import transforms, datasets, models


def getdatas(input_size,data_dir,batchSize,data_name):
    if data_name=='NEU-CLS':
        data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor()
            ]),
            'val': transforms.Compose([
                transforms.ToTensor()
            ]),
        }

        image_train=datasets.ImageFolder(os.path.join(data_dir,'train'),data_transforms['train']) 
        image_val=datasets.ImageFolder(os.path.join(data_dir,'val'),data_transforms['val']) 
        image_test=datasets.ImageFolder(os.path.join(data_dir,'val'),data_transforms['val']) 
        # dataloader 
        train_loader=torch.utils.data.DataLoader(image_train,batch_size=batchSize,shuffle=True, num_workers=4)
        val_loader=torch.utils.data.DataLoader(image_val,batch_size=batchSize,shuffle=False, num_workers=4)
        test_loader=torch.utils.data.DataLoader(image_test,batch_size=batchSize,shuffle=False, num_workers=4)
    return train_loader,val_loader,test_loader,image_train,image_val,image_test