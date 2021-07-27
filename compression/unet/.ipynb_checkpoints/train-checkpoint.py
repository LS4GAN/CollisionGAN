from pathlib import Path
import scipy.stats as ss
import os
import sys
import torch

from notunet import NotUNet

class dataset_toyzero(torch.utils.data.Dataset):
    """
    LS4GAN dataset
    """
    def __init__(
        self, 
        paths, *,
        clip=None,
        batch_size=1, 
        valid_fraction=.2,
        max_num=None,
    ):
        super(dataset_toyzero, self).__init__()    
        
        self.clip = clip
        
        self.image_fnames = []
        for path in paths:
            self.image_fnames += list(Path(path).glob('*npz'))
        self.image_fnames = np.array(self.image_fnames)
        
        indices = np.arange(len(self.image_fnames))
        np.random.shuffle(indices)
        if max_num is not None:
            indices = indices[:max_num]
        valid_size = int(len(indices) * valid_fraction)
        train_size = len(indices) - valid_size
        indices_train = indices[:train_size]
        indices_valid = indices[train_size:]
        print(f'train example: {train_size} / {len(indices)}')
        print(f'valid example: {valid_size} / {len(indices)}')

        self.data_loaders = {}
        for split, I in zip(['train', 'valid'], [indices_train, indices_valid]):
            datum = np.array(list(map(self._load_file, self.image_fnames[I])))
            loader = torch.utils.data.DataLoader(datum, batch_size=bsz, shuffle=True)
            self.data_loaders[split] = loader
    
    def _load_file(self, fname):
        datum = np.load(fname)
        datum = datum[datum.files[0]]
        if self.clip is not None:
            datum = datum[:self.clip[0], :self.clip[1]]
        mode = ss.mode(datum, axis=None)[0][0]
        datum -= mode
        return np.expand_dims(np.float32(datum), 0)
    
    def get_split(self, split):
        return self.data_loaders[split]

    def get_splits(self):
        return self.data_loaders['train'], self.data_loaders['valid']
    
    
def train(
    model, 
    train_loader, 
    valid_loader, 
    epochs,
    lr=1e-4,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    for e in range(epochs):
        train_loss = 0
        train_len = len(train_loader)
        for x in train_loader:
            x = x.cuda()
            pred = model(x)
            loss = loss_fn(pred, x)
            train_loss += loss.item()
            # back-propagate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= train_len

        valid_loss = 0
        valid_len = len(valid_loader)
        for x in valid_loader:
            x = x.cuda()
            pred = model(x)
            loss = loss_fn(pred, x)
            valid_loss += loss.item()
        valid_loss /= valid_len    
        scheduler.step(valid_loss)

        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break

        print(f'{e + 1}/{epochs}, train = {train_loss:.6f}, valid = {valid_loss:.6f}, lr={lr:.3e}')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Iterative training")

    parser.add_argument(
        '--model_pt_path', '-d',
        help    = 'Directory where toyzero dataset is located',
        type    = str,
    )

    parser.add_argument(
        '--block_begin', '-b',
        default = 2
        help    = 'the beginning number of blocks (inclusive)',
        type    = int,
    )

    parser.add_argument(
        '--block_end', '-e',
        default = 6,
        help    = 'the ending number of blocks (inclusive)',
        type    = int,
    )

    parser.add_argument(
        '--learning_rate', '-l',
        default = 1e-4,
        help    = 'learning rate',
        type    = float
    )
    
    parser.add_argument(
        '--epochs', '-c',
        default = 200,
        help    = 'number of epochs',
        type    = int
    )
    
    parser.add_argument(
        '--depth_seed', '-s',
        help    = 'the number channels (we are using constant here)',
        type    = int
    )
 
    args = parser.parse_args()
    
    
    bsz = 20
    paths = [
        Path(f'/sdcc/u/yhuang2/PROJs/GAN/datasets/ls4gan/toyzero/toyzero_2021-06-29_U/trainA/'),
        Path(f'/sdcc/u/yhuang2/PROJs/GAN/datasets/ls4gan/toyzero/toyzero_2021-06-29_U/trainB/')
    ]


    ds = dataset_toyzero(paths, clip=[128, 128], batch_size=bsz, valid_fraction=.4, max_num=2000)
    train_loader, valid_loader = ds.get_splits()
    print(f'Number of batches:\
        \n\ttrain = {len(train_loader)}\
        \n\tvalidation = {len(valid_loader)}')
    
    
    # Training iteratively
    depth_seed = args.depth_seed
    starting_blocks = args.block_begin
    blocks = args.block_end
    epochs = args.epochs
    loss_fn = torch.nn.L1Loss(reduction='mean')

    prev_model = NotUNet(		
        input_channels=1, 
        depth_seed=depth_seed, 
        blocks=starting_blocks,
        growth='constant', 
        activation='relu'
    ).cuda()

    # Train the previous model:
    print(f'Training model {starting_blocks}')
    train(prev_model, train_loader, valid_loader, epochs, lr=1e-4)
    pt_fname = f'{model_pt_folder}/notunet_{depth_seed}_{starting_blocks}.pt'
    torch.save(prev_model.state_dict(), pt_fname)


    for b in range(starting_blocks + 1, blocks + 1):
        transferable_params = prev_model.get_transferable_params()

        # get a model of 1 more block
        model = NotUNet(		
            input_channels=1, 
            depth_seed=depth_seed, 
            blocks=b,
            growth='constant', 
            activation='relu'
        ).cuda()

        # load transferable parameters
        model.load_transferable_params(transferable_params)

        # Train the newly added blocks while freezing the loaded blocks 
        print(f'\nPretrain training model {b}, newly added blocks')
        model.make_lower_subnetwork_trainable(False)
        train(model, train_loader, valid_loader, int(epochs // 4))

        # make all layers trainable
        print(f'\nTraining full model {b}')
        model.make_lower_subnetwork_trainable(True)
        train(model, train_loader, valid_loader, epochs, lr=1e-4)

        pt_fname = f'{model_pt_folder}/notunet_{depth_seed}_{b}.pt'
        torch.save(model.state_dict(), pt_fname)

        prev_model = model