import copy
import numpy as np
from pathlib import Path
import polars as pl
import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import tqdm.auto as tqdm
import wandb

from deep_parity.boolean_cube import fourier_transform, generate_all_binary_arrays
from deep_parity.model import MLP


def get_activations(model, n, path):
    batch_size = 2 ** 14
    bits = torch.from_numpy(generate_all_binary_arrays(n)).to(torch.float32)
    activations = []
    for batch in bits.split(batch_size):
        _, cache = model.run_with_cache(batch.to('cuda'))
        activations.append(cache[path].detach().cpu())
    return torch.concatenate(activations)


def make_base_parity_dataframe(n):
    all_binary_data_zero_one = generate_all_binary_arrays(n)
    all_binary_data = np.sign(-1 * all_binary_data_zero_one + 0.5)
    all_parities = all_binary_data.prod(axis=1)
    base_df = pl.DataFrame({
        'bits': all_binary_data, 
        'parities': all_parities, 
    })
    base_df = base_df.with_columns(
        indices=pl.col('bits').arr.to_list().list.eval(pl.arg_where(pl.element() == -1))
    )
    return base_df


def calc_power_contributions(tensor, n, epoch):
    linear_dim = tensor.shape[1]
    base_df = make_base_parity_dataframe(n)
    ft = fourier_transform(tensor.T.to('cuda'))
    ft /= ft.mean(dim=0, keepdims=True)
    linear_df = pl.DataFrame(
        ft.T.detach().cpu().numpy(),
        schema=[str(i) for i in range(linear_dim)]
    )
    data = pl.concat([base_df, linear_df], how='horizontal')
    total_power = (
        data
        .select(pl.exclude('bits', 'parities', 'indices', 'degree'))
        .unpivot()
        .with_columns(pl.col('variable').str.to_integer())
        .group_by('variable').agg(pl.col('value').pow(2).sum())
        .rename({'value': 'power'})
    )
    power_df = (
        data
        .select(pl.exclude('bits', 'parities', 'indices'))
        .unpivot(index='degree')
        .with_columns(pl.col('variable').str.to_integer())
        .group_by('degree', 'variable').agg(pl.col('value').pow(2).sum())
        .join(total_power, on='variable', how='left')
        .with_columns(pcnt_power = pl.col('value') / pl.col('power'), epoch=pl.lit(epoch))
    )
    return power_df


def fourier_analysis(model, n, epoch):
    model.eval()
    with torch.no_grad():
        linear_preacts = get_activations(model, n, 'hook_linear2')
    embed_power_df = calc_power_contributions(linear_preacts, n, epoch)
    model.train()
    return embed_power_df


def get_dataloaders(n, batch_size, frac_train, seed):
    sequences = torch.from_numpy(generate_all_binary_arrays(n)).to(torch.float32)
    parities = sequences.sum(dim=1).to(torch.int64) % 2
    data = TensorDataset(sequences, parities)
    train_data, test_data = random_split(
        data,
        [frac_train, 1 - frac_train],
        torch.Generator().manual_seed(seed)
    )
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    return train_dataloader, test_dataloader


def loss_fn(logits, labels):
    if len(logits.shape) == 3:
        logits = logits[:, -1]
    logits = logits.to(torch.float64)
    log_probs = logits.log_softmax(dim=-1)
    correct_log_probs = log_probs.gather(dim=-1, index=labels.unsqueeze(1))[:, 0]
    return (-1. * correct_log_probs).mean()


def train_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda')
    for bits, parities in dataloader:
        logits = model(bits.to('cuda'))
        losses = loss_fn(logits, parities.to('cuda'))
        mean_loss = losses.mean()
        mean_loss.backward()
        total_loss += mean_loss
    return total_loss.item()


def test_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda')
    num_batches = 2
    i = 0
    for bits, parities in dataloader:
        if i > num_batches:
            break
        logits  = model(bits.to('cuda'))
        losses = loss_fn(logits, parities.to('cuda'))
        total_loss += losses.mean()
    return total_loss.item()


def endless_data_loader(data):
    while True:
        for batch in data:
            yield batch
        

def train(model, optimizer, train_dataloader, test_dataloader, config, seed):
    train_config = config['train']
    n = config['model']['n']
    checkpoint_dir = Path('checkpoints-wd01')
    #checkpoint_dir = setup_checkpointing(train_config, seed)
    model_checkpoints = []
    opt_checkpoints = []
    train_loss_data = []
    test_loss_data = []

    #train_data = endless_data_loader(train_dataloader)
    #test_data = endless_data_loader(test_dataloader)

    for step in tqdm.tqdm(range(train_config['num_steps'])):
        #bits, parities = next(train_data)
        #logits = model(bits.to('cuda'))
        train_loss = train_forward(model, train_dataloader)
        optimizer.step()
        optimizer.zero_grad()

        msg = {'loss/train': train_loss}

        model.eval()
        with torch.no_grad():
            #test_bits, test_parities = next(test_data)
            #test_logits = model(test_bits.to('cuda'))
            test_loss = test_forward(model, test_dataloader)
            msg['loss/test'] = test_loss
        model.train()

        optimizer.zero_grad()
        
        if step % 200 == 0:
            linear_data = fourier_analysis(model, n, step)
            df = linear_data.group_by('degree').agg(pl.col('pcnt_power').implode())
            linear_powers = {f"linear/degree{int(rec['degree'])}": rec['pcnt_power'][0] for rec in df.to_dicts()}
            msg.update(linear_powers)
           
        if step % 10_000 == 0:
            train_loss_data.append(train_loss)
            test_loss_data.append(test_loss)
            model_state = copy.deepcopy(model.state_dict())
            opt_state = copy.deepcopy(optimizer.state_dict())

            torch.save(
                {
                    "model": model_state,
                    "optimizer": opt_state,
                    "config": config['model'],
                    "rng": torch.get_rng_state()
                },
                checkpoint_dir / f'{step}.pth'
            )
            model_checkpoints.append(model_state)
            opt_checkpoints.append(opt_state)
        
        wandb.log(msg)


    torch.save(
        {
            "model": model.state_dict(),
            "config": config['model'],
            "checkpoints": model_checkpoints,
        },
        checkpoint_dir / "full_run.pth"
    )

def main():
    ###########################
    # Configs
    ###########################
    n = 18
    batch_size = 2 ** 16
    frac_train = 0.95
    embed_dim = 1024
    model_dim = 1024
    optimizer_params = {
        "lr" : 1e-4,
        "weight_decay" : 0.1,
        "betas" : [0.9, 0.98]
    }
    num_steps = 50_000
    device = torch.device('cuda')
    seed = 3141529
    #############################


    torch.manual_seed(seed)

    train_data, test_data = get_dataloaders(
        n,
        batch_size,
        frac_train,
        seed
    )

    checkpoint_dir = Path('checkpoints-wd01')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
 

    model = MLP(n, embed_dim, model_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **optimizer_params
    )

    config = {
        "model": {
            "n": n,
            "embed_dim": embed_dim,
            "model_dim": model_dim,
        },
        "optim": optimizer_params,
        "train": {
            "batch_size": batch_size,
            "frac_train": frac_train,
            "num_steps": num_steps
        }
    }

    wandb.init(
        entity='dstander',
        group="parity-3Layer",
        project="deep-parity",
        config=config
    )

    wandb.watch(model, log="all", log_freq=200)

    try:
        train(
            model,
            optimizer,
            train_data,
            test_data,
            config,
            seed
        )
    except KeyboardInterrupt:
        pass

    wandb.finish()


if __name__ == '__main__':
    main()

