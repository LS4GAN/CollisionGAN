#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# In[2]:


from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Model and trial

# In[3]:


class AlexNetCAMD(nn.Module):
    def __init__(self, input_channels=3, **kwargs):
        super(AlexNetCAMD, self).__init__(**kwargs)
        self.base_net = nn.Sequential(
            nn.BatchNorm2d(input_channels),

            nn.Conv2d(input_channels, 96, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(384, 2)

    def forward(self, x):
        y = self.base_net(x)
        z = torch.mean(y, dim=(2, 3), keepdim=False) # Global average
        return self.classifier(z)

    def get_cam(self, x):
        y = self.base_net(x)
        maps = []
        for class_weights in self.classifier.weight:
            # y             : (N, C_o, W_o, H_o)
            # class_weights : (C_o, )
            # class_map     : (N, W_o, H_o)
            class_map = torch.tensordot(y, class_weights, dims=([1,], [0,]))

            # class_map : (N, W_o, H_o) -> (N, 1, W_i, H_i)
            class_map = nn.functional.interpolate(
                torch.unsqueeze(class_map, 1),
                (x.shape[2], x.shape[3]),
                mode='bilinear'
            )
            maps.append(class_map)

        return maps


# In[4]:


def trial(model, input_shape, cuda=True):
    x = torch.rand(16, 1, *input_shape, dtype=torch.float32)
    if cuda:
        model = model.cuda()
        x = x.cuda()

    y = model.forward(x)
    print(f'Output shape:\n\t{y.shape}\n')

    num_parameters_trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    num_parameters = sum([p.numel() for p in model.parameters()])
    print(f'Number of parameters:\n\tTrainable = {num_parameters_trainable}\n\tTotal = {num_parameters}') 


# In[5]:


discriminator = AlexNetCAMD(input_channels=1)
input_shape = [128, 128]
trial(discriminator, input_shape)


# ## Load data

# In[6]:


# class Dataset_ls4gan(Dataset):
#     """
#     LS4GAN dataset
#     """
#     def __init__(self, class_paths, num_samples=None):
#         super(Dataset_ls4gan, self).__init__()    
#         self.image_fnames = []
#         self.labels = []
#         for c, class_path in enumerate(class_paths):
#             fnames = list(Path(class_path).glob('*npz'))
#             self.image_fnames += fnames
#             self.labels += [c] * len(fnames)
#         indices = np.arange(len(self.image_fnames))
#         np.random.shuffle(indices)
#         if num_samples is not None:
#             indices = indices[:num_samples]
        
#         self.image_fnames = np.array(self.image_fnames)[indices]
#         self.labels = np.array(self.labels)[indices]
        
#     def __len__(self):
#         return len(self.image_fnames)
    
#     def __getitem__(self, idx):
#         image_fname, label = self.image_fnames[idx], self.labels[idx]
        
#         image = np.load(image_fname)
#         key = list(image.keys())[0]
#         image = image[key]
#         image = np.expand_dims(np.float32(image), 0)
        
#         image_tensor = torch.from_numpy(image)
#         label_tensor = torch.tensor(label, dtype=torch.int64)
#         return image_tensor, label_tensor


# path_base = '/sdcc/u/yhuang2/PROJs/GAN/datasets/ls4gan/toyzero_cropped'
# dataset = 'toyzero_2021-06-29_safi_'

# layer = 'W'

# class_paths_train = [
#     f'{path_base}/{dataset}{layer}/trainA/',
#     f'{path_base}/{dataset}{layer}/trainB/'
# ]

# class_paths_test = [
#     f'{path_base}/{dataset}{layer}/testA/',
#     f'{path_base}/{dataset}{layer}/testB/'
# ]


# num_samples, bsz = 2000, 16

# dataset_train = Dataset_ls4gan(class_paths_train, num_samples=num_samples)
# dataset_test = Dataset_ls4gan(class_paths_test, num_samples=num_samples)
# dataset_test_d = Dataset_ls4gan(class_paths_test_d)

# train_loader = DataLoader(dataset_train, batch_size=bsz, shuffle=True)
# test_loader = DataLoader(dataset_test, batch_size=bsz, shuffle=True)
# test_loader_d = DataLoader(dataset_test_d, batch_size=bsz, shuffle=True)


# In[7]:


class Dataset_ls4gan(Dataset):
    """
    LS4GAN dataset
    """
    def __init__(
        self, 
        data_path, 
        window_fname, 
        num_samples=None,
        apa=None,   # If not None, must be list 
        planes=None, # If not None, must be list, too,
        valid_fraction=.2,
        batch_size=32,
    ):
        super(Dataset_ls4gan, self).__init__()
        df_window = pd.read_csv(window_fname, index_col=0)
        
        # select
        if apa is not None:
            df_window = df_window[df_window.apa.isin(apa)]
        if planes is not None:
            df_window = df_window[df_window.plane.isin(planes)]
        if num_samples is not None and num_samples < len(df_window):
            df_window = df_window.sample(n=num_samples // 2 , replace=False).reset_index(drop=True)
            
        # check existence of folders
        assert Path(data_path).exists(), f"{data_path} doesn't exist"
        fake_path = Path(data_path)/'fake'
        real_path = Path(data_path)/'real'
        assert fake_path.exists(), f"{data_path} doesn't contain a subfolder called fake"
        assert fake_path.exists(), f"{data_path} doesn't contain a subfolder called real"
        
        # Load data
        data_train, data_valid = [], []
        for P, c in zip([fake_path, real_path], [0, 1]):
            print(f'loading files from {P}')
            for index, row in df_window.iterrows():
                image, bkg = row['image'], row['bkg']
                x, y, x_ws, y_ws = row['x'], row['y'], row['width'], row['height']
                image_fname = P/image
                image = self._load_image(image_fname, x, y, x_ws, y_ws, bkg)
                
                rnd_key = np.random.rand()
                if rnd_key < valid_fraction:
                    data_valid.append([image, c])
                else:
                    data_train.append([image, c])

        self.loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)
        self.loader_valid = DataLoader(data_valid, batch_size=batch_size, shuffle=True)
        print('Done!')
    
    def _load_image(self, image_fname, x, y, x_ws, y_ws, bkg):
        image = np.load(image_fname)
        key = list(image.keys())[0]
        image = image[key]
        image = image[x: x + x_ws, y: y + y_ws]
        image -= bkg
        image = np.expand_dims(np.float32(image), 0)
        
        return image   
        
    def get_loaders(self):
        return self.loader_train, self.loader_valid


# In[54]:


data_path = '/hpcgpfs01/scratch/yhuang2/merged'
min_signal = 250
window_fname = f'/hpcgpfs01/scratch/yhuang2/merged/windows_{min_signal}-128x128.csv'
batch_size = 32
plane = 'U'
dl = Dataset_ls4gan(data_path, window_fname, num_samples=2000, planes=[plane], batch_size=batch_size)
train_loader, test_loader = dl.get_loaders()


# ## Train

# In[55]:


# discriminator = AlexNetCAMD(input_channels=1).cuda()
# discriminator.load_state_dict(torch.load(f'results/model_dict_{layer}.pt'))
# discriminator.eval()


# In[56]:


discriminator = AlexNetCAMD(input_channels=1).cuda()

criterion = nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.5, patience=10)

def calc_correct(pred, true_class):
    pred_class = torch.argmax(pred, dim=1, keepdim=False)
    result = torch.sum(pred_class == true_class, dim=0)
    return result

epochs = 200

for epoch in range(epochs):  # loop over the dataset multiple times

    train_loss, num_correct, total = 0, 0, 0
    for i, data in enumerate(train_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = discriminator(inputs)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        num_correct += calc_correct(outputs, labels)
        total += len(outputs)
        # print(num_correct, total)
    
    train_loss_avg = train_loss / len(train_loader)
    acc = num_correct / total
    print(f'\nEpoch: {epoch + 1} / {epochs}')
    print(f'\ttrain:\tloss = {train_loss_avg:.6f}, acc = {acc:.6f}')
    
    test_loss, num_correct, total = 0, 0, 0
    for i, data in enumerate(test_loader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()

        # forward + backward + optimize
        outputs = discriminator(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        num_correct += calc_correct(outputs, labels)
        total += len(outputs)
        
    test_loss_avg = test_loss / len(test_loader)
    acc = num_correct / total
    print(f'\ttest:\tloss = {test_loss_avg:.6f}, acc = {acc:.6f}')
    
    scheduler.step(test_loss_avg)
    
    for param_group in optimizer.param_groups:
        cur_lr = param_group['lr']
        break
    print(f'\tlr={cur_lr:.2e}')

print('Finished Training')

dataset = f'rnd_crop_{min_signal}'
torch.save(discriminator.state_dict(), f'results/{dataset}_model_dict_{plane}.pt')


# ## Evaluation
# Make sure to run evaluation after all planes are done!

# In[42]:


# df_data = []
# wires = ['U', 'V', 'W']
# for layer in wires:
#     print(f'{layer}:')
    
#     # Load data
#     class_paths_train = [f'{path_base}/{dataset}{layer}/trainA/', f'{path_base}/{dataset}{layer}/trainB/']
#     class_paths_test = [f'{path_base}/{dataset}{layer}/testA/', f'{path_base}/{dataset}{layer}/testB/']
#     class_paths_test_d = [f'{path_base}/Dmitrii/{layer}/testA/', f'{path_base}/Dmitrii/{layer}/testB/']

#     dataset_train = Dataset_ls4gan(class_paths_train)
#     dataset_test = Dataset_ls4gan(class_paths_test)
#     dataset_test_d = Dataset_ls4gan(class_paths_test_d)

#     train_loader = DataLoader(dataset_train, batch_size=bsz, shuffle=True)
#     test_loader = DataLoader(dataset_test, batch_size=bsz, shuffle=True)
#     test_loader_d = DataLoader(dataset_test_d, batch_size=bsz, shuffle=True)
    
#     # Load model
#     discriminator = AlexNetCAMD(input_channels=1).cuda()
#     discriminator.load_state_dict(torch.load(f'results/{dataset}model_dict_{layer}.pt'))
#     discriminator.eval()
    
#     # Evaluation
#     splits = ['train', 'test', 'test_d']
#     df_data_row = []
#     with torch.no_grad():
#         for split, loader in zip(splits, [train_loader, test_loader, test_loader_d]):
#             total_example, total_correct = 0, 0
#             for i, data in enumerate(loader):
#                 inputs, labels = data
#                 inputs = inputs.cuda()
#                 labels = labels.cuda()
#                 pred = discriminator(inputs)

#                 pred_class = torch.argmax(pred, dim=1, keepdim=False)
#                 result = torch.sum(pred_class == labels, dim=0)
#                 total_correct += result
#                 total_example += labels.shape[0]

#             acc = total_correct / total_example
#             print(f'\t{split} accuracy = {acc:.3f}')
#             df_data_row.append(acc.cpu().detach().numpy())
#     df_data.append(df_data_row)

# df_result = pd.DataFrame(data=df_data, columns=['train', 'test', 'test_d'], index=wires)
# df_result


# In[52]:


df_data = []
planes = ['U', 'V', 'W']
for plane in planes:
    print(f'{plane}:')

    dl = Dataset_ls4gan(
        data_path, 
        window_fname, 
        num_samples=2000, 
        planes=[plane], 
        batch_size=batch_size)
    train_loader, test_loader = dl.get_loaders()
    
    # Load model
    discriminator = AlexNetCAMD(input_channels=1).cuda()
    discriminator.load_state_dict(torch.load(f'results/{dataset}_model_dict_{plane}.pt'))
    discriminator.eval()
    
    
    splits = ['train', 'test']
    loaders = [train_loader, test_loader]
    
    df_data_row = []
    with torch.no_grad():
        for split, loader in zip(splits, loaders):
            total_example, total_correct = 0, 0
            for i, data in enumerate(loader):
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()
                pred = discriminator(inputs)

                pred_class = torch.argmax(pred, dim=1, keepdim=False)
                result = torch.sum(pred_class == labels, dim=0)
                total_correct += result
                total_example += labels.shape[0]

            acc = total_correct / total_example
            print(f'\t{split} accuracy = {acc:.3f}')
            df_data_row.append(acc.cpu().detach().numpy())
    df_data.append(df_data_row)

df_result = pd.DataFrame(data=df_data, columns=['train', 'test'], index=planes)    
df_result.to_csv(f'results/{dataset}_ACC.csv', float_format='%.4f')


# In[53]:


dfs = []
for fname in Path('results').glob('rnd_crop_*_ACC.csv'):
    min_signal = int(fname.stem.split('_')[-2])
    df = pd.read_csv(fname, index_col=0)
    df[f'{min_signal}'] = (df['train'] + df['test']) / 2
    dfs.append(df[f'{min_signal}'])
df = pd.concat(dfs, axis=1)
df = df.reindex(sorted(df.columns, key=lambda x: int(x)), axis=1)
df


# In[ ]:




