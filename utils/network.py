import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler

import random


class Identity(nn.Module):
    def forward(self, x):
        return x
    
def get_norm_layer(norm_type='instance'):    
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    elif norm_type == 'none':
        norm_layer = Identity()
    else:
        raise NotImplementedError(f'normalization layer {norm_type} is not found')
    
    return norm_layer


def init_weights(net, init_gain=0.02):
    """
    Let us keep it simple and only use normal initialization
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') !=-1):
            init.normal_(m.weight.data, 0.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
    net.apply(init_func)
    print('initializae network with normal distribution')
    
    
def get_scheduler(
    optimizer, *,
    lr_policy='linear', 
    n_epochs_no_decay=100, 
    n_epochs_decay=100,
    lr_decay_steps=50
):
    """
    Retrun a learning rate scheduler
    
    Parameters:
        1. optimizer (nn.optim): the optimizer
        2. lr_policy: the name of learning rate policy: linear | step | plateau | cosin
        
    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    
    For other schedulers (step, plateau, and consine), we use the default PyTorch shedulers.
    See https://pytorch.org/docs/stable/optim.html for more detail
    """
    
    if lr_policy == 'linear':
        def lr_lambda(epoch):
            return 1. - max(0, epoch - n_epochs_no_decay) / float(n_epochs_decay + 1)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif lr_policy == 'step':
        # multiply by gamma every lr_decay_steps
        # for example lr_decay_steps=50 and initial learning = .5
        # then we have 
        #     lr = .5 for 0 <= epoch < 50;
        #     lr = .05 for 50 <= epoch < 100;
        #     lr = .005 for 100 <= epoch < 150;
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_steps, gamma=.1)
    elif lr_policy == 'plateau':
        # Reduce learning rate when a metric has stopped improving. 
        # Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. 
        # This scheduler reads a metrics quantity and if no improvement 
        # is seen for a ‘patience’ number of epochs, 
        # the learning rate is reduced.
        # Parameters
        #    - mode (str, default=min): In `min` mode, lr will be reduced when the quantity monitored has stopped decreasing; 
        #        in `max` mode, lr will be reduced when the quantity monitored has stopped increasing.
        #    - factor (float, default=.1): Factor by which the learning rate will be reduced. new_lr = lr * factor.
        #    - patience (int, default=10): Number of epochs with no improvement after which learning rate will be reduced. 
        #    - threshold (float): only decrease lr if the change in the quantitiy monitored is smaller than threshold. 
        #        Say we have threshold=0.001, if loss is $18.0$ on epoch $n$ and loss is $17.9999$ on epoch $n+1$,
        #        then multiply current learning rate by the factor.
        #        On the contrary, if the loss is 17.99, lr doesn't have to be changed.
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.2, threshold=.01, patience=5)
    elif lr_policy == 'cosine':
        # make cosinusoidal change between η_min and η_max = initial_learning_rate every T_max epochs
        # lr_t = η_min + .5(η_max − η_min)(1 + cos(π * T_cur / T_max))
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epoch_no_decay, eta_min=0)
    else:
        return NotImplementedError(f'learning rate policy {lr_policy} is not implemented')
    return scheduler


class ImagePool():
    """
    This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than just the ones just produced.
    """

    def __init__(self, pool_size, p=.5):
        """
        Initialize the ImagePool class
        
        Parameters:
            1. pool_size (int): the size of image buffer.
            2. the fraction of images from the buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_images = 0
            self.images = []
            self.p = p
        
    def query(self, images):
        """
        Parameters:
            1. images: the latest generated images
            
        Return:
            len(images) images, a mixture from images and the buffer.
        """
        
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if  self.num_images < self.pool_size:
                # when the pool is not full, fill in the pool 
                # with newly generated image
                # and only return newly generated image.
                self.num_images += 1
                self.images.append(image)
                return_images.append(image)
            else:
                if random.uniform(0, 1) < p:
                    # with chance p, return a historical image and 
                    # use newly generated image to fill the vacancy
                    random_id = random.randint(0, self.pool_size - 1)
                    historical_image = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(historical_image)
                else:
                    # with chance 1 - p return a newly generated image.
                    return_images.append(image)
                    
        return torch.cat(return_images, 0)