import torch
import copy

from config   import Config, get_save_dir
from data     import get_data
from model    import check_model, get_model, save_model, load_model
from parsers  import parse_train_cmdargs
from nn_funcs import train, get_torch_device_smart, seed_everything

def get_label(config, base_label):
    return '%s-%s' % (base_label, config.model_args['blocks'])

def construct_prev_config(config):
    result = copy.deepcopy(config)
    result.model_args['blocks'] = config.model_args['blocks'] - 1

    return result

def train_model(model, config, base_label, it_train, it_val, device, loss_fn):
    seed_everything(config.seed)

    optimizer = torch.optim.Adam(
        model.parameters(), lr = config.lr, weight_decay = 1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    history = train(
        it_train, it_val, model, loss_fn, optimizer, scheduler,
        config.epochs, config.steps_per_epoch, device
    )

    save_model(model, config, history, get_label(config, base_label))

def pretrain_model(
    model, config, base_label, it_train, it_val, device, loss_fn
):
    seed_everything(config.seed)

    optimizer = torch.optim.Adam(
        model.parameters(), lr = config.lr, weight_decay = 1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train(
        it_train, it_val, model, loss_fn, optimizer, scheduler,
        config.epochs // 4, config.steps_per_epoch, device
    )

def recursive_train(config, base_label, it_train, it_val, device, loss_fn):
    savedir = get_save_dir(config, get_label(config, base_label))
    blocks  = config.model_args['blocks']

    if check_model(savedir):
        _, model = load_model(savedir, device)
        print("Loaded model with %d blocks." % blocks)
        return model

    print("Model with %d blocks is not found." % blocks)
    model = get_model(config.model, device, config.model_args)
    model.train()

    if blocks == 0:
        print("Training model with 0 blocks...")
        train_model(
            model, config, base_label, it_train, it_val, device, loss_fn
        )
        return model

    prev_config = construct_prev_config(config)
    prev_model  = recursive_train(
        prev_config, base_label, it_train, it_val, device, loss_fn
    )

    print(
        "Base model trained. Transferring parameters to model with %d blocks"
            % blocks
    )
    transferable_params = prev_model.get_transferable_params()
    model.load_transferable_params(transferable_params)

    model.make_lower_subnetwork_trainable(False)

    print("Pretraining model with %d blocks..." % blocks)
    pretrain_model(
        model, config, base_label, it_train, it_val, device, loss_fn,
    )

    model.make_lower_subnetwork_trainable(True)

    print("Training model with %d blocks..." % blocks)
    train_model(
        model, config, base_label, it_train, it_val, device, loss_fn
    )

    return model

def main(config, cmdargs):
    seed_everything(config.seed)

    base_label = cmdargs.label
    device     = get_torch_device_smart()
    loss_fn    = torch.nn.L1Loss(reduction = 'mean')

    it_train, it_val = \
        get_data(config.data, config.data_args, config.batch_size)

    recursive_train(config, base_label, it_train, it_val, device, loss_fn)

if __name__ == '__main__':
    config, cmdargs = parse_train_cmdargs()
    main(config, cmdargs)