import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, IterableDataset
import tqdm.auto as tqdm
import wandb

from deep_parity.model import MLP


def generate_random_bits(batch_size, n):
    return 2 * torch.randint(0, 2, (batch_size, n), dtype=torch.float32) - 1


def calculate_k_sparse_parity(bits, k_indices):
    return ((bits[:, k_indices].prod(dim=1) + 1) / 2).long()


def get_dataloaders(batch_size, n, k_indices):
    def data_generator():
        while True:
            bits = generate_random_bits(batch_size, n)
            parities = calculate_k_sparse_parity(bits, k_indices)
            yield bits, parities

    #train_dataloader = DataLoader(data_generator(), batch_size=None)
    #test_dataloader = DataLoader(data_generator(), batch_size=None)

    return data_generator(), data_generator()


def estimate_prediction_variance(model, n, k, k_indices, num_samples=1000):
    device = next(model.parameters()).device
    fixed_k_bits = 2 * torch.randint(0, 2, (1, k), device=device, dtype=torch.float32) - 1
    
    # Generate full batches of n bits
    full_inputs = generate_random_bits(num_samples, n).to(device)
    
    # Overwrite the k bits with the fixed set
    full_inputs[:, k_indices] = fixed_k_bits.repeat(num_samples, 1)
    
    with torch.no_grad():
        logits = model(full_inputs)
        probs = torch.softmax(logits, dim=-1)
        probabilities = probs[:, 1].cpu().numpy()  # Probability of parity being 1
    
    return np.var(probabilities)


def train_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda')
    num_batches = 10
    for _ in range(num_batches):
        bits, parities = next(dataloader)
        logits = model(bits.to('cuda'))
        losses = torch.nn.functional.cross_entropy(logits, parities.to('cuda'))
        mean_loss = losses.mean()
        mean_loss.backward()
        total_loss += mean_loss
    return total_loss.item()


def test_forward(model, dataloader):
    total_loss = torch.tensor(0., device='cuda')
    num_batches = 2
    for _ in range(num_batches):
        bits, parities = next(dataloader)
        with torch.no_grad():
            logits = model(bits.to('cuda'))
            losses = torch.nn.functional.cross_entropy(logits, parities.to('cuda'))
        total_loss += losses.mean()
    return (total_loss / num_batches).item()


def train(model, optimizer, train_dataloader, test_dataloader, config, k_indices):
    train_config = config['train']
    n = config['model']['n']
    k = config['model']['k']
    checkpoint_dir = Path('checkpoints-n20-k10-signed')
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    for step in tqdm.tqdm(range(train_config['num_steps'])):
        train_loss = train_forward(model, train_dataloader)
        optimizer.step()
        optimizer.zero_grad()

        msg = {'loss/train': train_loss}

        if step % 100 == 0:
            test_loss = test_forward(model, test_dataloader)
            msg['loss/test'] = test_loss
            prediction_variance = estimate_prediction_variance(model, n, k, k_indices)
            msg['prediction_variance'] = prediction_variance

        if step % 10000 == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config['model'],
                    "step": step,
                },
                checkpoint_dir / f'checkpoint_{step}.pth'
            )

        wandb.log(msg)

    torch.save(
        {
            "model": model.state_dict(),
            "config": config['model'],
        },
        checkpoint_dir / "final_model.pth"
    )

def main():
    ###########################
    # Configs
    ###########################
    n = 30
    k = 10
    k_indices = np.random.choice(n, k, replace=False)
    batch_size = 2 ** 16
    layer_sizes = [2048, 4096, 2048]
    optimizer_params = {
        "lr": 3e-4,
        "weight_decay": 0.01,
        "betas": [0.9, 0.98]
    }
    num_steps = 1_00_000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')
    seed = 3141529
    #############################

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data, test_data = get_dataloaders(batch_size, n, k_indices)

    model = MLP(n, layer_sizes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **optimizer_params
    )

    config = {
        "model": {
            "n": n,
            "k": k,
            "k_indices": k_indices.tolist(),
            "layer_sizes": layer_sizes
        },
        "optim": optimizer_params,
        "train": {
            "batch_size": batch_size,
            "num_steps": num_steps
        }
    }

    wandb.init(
        entity='dstander',
        group="k-sparse-parity-n20-k10-signed",
        project="deep-parity",
        config=config
    )

    wandb.watch(model, log="all", log_freq=1000)

    try:
        train(
            model,
            optimizer,
            train_data,
            test_data,
            config,
            k_indices
        )
    except KeyboardInterrupt:
        pass

    wandb.finish()

if __name__ == '__main__':
    main()
