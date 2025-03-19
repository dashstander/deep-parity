import copy
import numpy as np
from pathlib import Path
import polars as pl
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import equinox as eqx
import optax
from functools import partial
import tqdm.auto as tqdm
import wandb
from google.cloud import storage

from deep_parity.jax.boolean_cube import fourier_transform, generate_all_binary_arrays
from deep_parity.jax.model import Perceptron


def get_activations(model, params, n):
    batch_size = 2 ** 14
    bits = jnp.array(generate_all_binary_arrays(n), dtype=jnp.float32)
    activations = []
    
    # Split into batches
    num_batches = (bits.shape[0] + batch_size - 1) // batch_size
    
    @jit
    def get_activations_batch(batch):
        # Extract the linear layer activations
        return jax.nn.relu(model.apply_linear(params, batch))
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, bits.shape[0])
        batch = bits[start_idx:end_idx]
        acts = get_activations_batch(batch)
        activations.append(acts)
    
    return jnp.concatenate(activations)


def make_base_parity_dataframe(n):
    all_binary_data = generate_all_binary_arrays(n).astype(np.int32)
    all_parities = all_binary_data.sum(axis=1) % 2
    base_df = pl.DataFrame({
        'bits': all_binary_data, 
        'parities': all_parities, 
    })
    base_df = base_df.with_columns(
        indices=pl.col('bits').arr.to_list().list.eval(pl.arg_where(pl.element() == 1)),
        degree=pl.col('bits').arr.sum().cast(pl.Int32),
    )
    return base_df


def calc_power_contributions(tensor, n, epoch):
    linear_dim = tensor.shape[1]
    base_df = make_base_parity_dataframe(n)
    centered_tensor = tensor - jnp.mean(tensor, axis=0, keepdims=True)
    ft = fourier_transform(centered_tensor.T)
    linear_df = pl.DataFrame(
        np.array(ft.T),  # Convert JAX array to numpy
        schema=[str(i) for i in range(linear_dim)]
    )
    data = pl.concat([base_df, linear_df], how='horizontal')
    total_power = (
        data
        .select(pl.exclude('bits', 'parities', 'indices', 'degree'))
        .unpivot()
        .with_columns(pl.col('variable').str.to_integer())
        .group_by(['variable']).agg(pl.col('value').pow(2).sum())
        .rename({'value': 'power'})
    )
    powers = {}
    for i in range(1, n):
        power_df = (
            data.filter(pl.col('degree') == i)
            .select(pl.exclude('bits', 'parities', 'indices', 'degree'))
            .unpivot()
            .with_columns(pl.col('variable').str.to_integer())
            .group_by(['variable']).agg(pl.col('value').pow(2).sum())
            .join(total_power, on='variable', how='left')
            .with_columns(pcnt_power = pl.col('value') / pl.col('power'), epoch=pl.lit(epoch))
            .sort('variable')
            .filter(pl.col('pcnt_power').is_not_nan())
        )
        powers[f'degree_{i}'] = power_df['pcnt_power'].to_numpy()
    return powers


def fourier_analysis(model, params, n, epoch):
    linear_preacts = get_activations(model, params, n)
    embed_power_df = calc_power_contributions(linear_preacts, n, epoch)
    return embed_power_df


def get_datasets(n, frac_train, seed):
    key = jax.random.PRNGKey(seed)
    sequences = jnp.array(generate_all_binary_arrays(n), dtype=jnp.float32)
    sequences = -1. * jnp.sign(sequences - 0.5)
    parities = -1 * ((jnp.prod(sequences, axis=1) - 1) / 2).astype(jnp.int32)
    
    # Shuffle the data
    shuffle_key, key = jax.random.split(key)
    indices = jax.random.permutation(shuffle_key, sequences.shape[0])
    sequences = sequences[indices]
    parities = parities[indices]
    
    # Split into train and test
    split_idx = int(frac_train * sequences.shape[0])
    train_sequences, test_sequences = sequences[:split_idx], sequences[split_idx:]
    train_parities, test_parities = parities[:split_idx], parities[split_idx:]
    
    return (train_sequences, train_parities), (test_sequences, test_parities)


def get_batch(data, batch_size, step):
    sequences, parities = data
    idx = (step * batch_size) % (sequences.shape[0] - batch_size)
    batch_sequences = sequences[idx:idx + batch_size]
    batch_parities = parities[idx:idx + batch_size]
    return batch_sequences, batch_parities


def loss_fn(params, model, bits, labels):
    logits = model.apply(params, bits)
    if len(logits.shape) == 3:
        logits = logits[:, -1]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    one_hot_labels = jax.nn.one_hot(labels, 2)
    return -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=-1))


@jit
def train_step(params, opt_state, model, batch):
    bits, labels = batch
    
    def compute_loss_and_grads(params):
        loss = loss_fn(params, model, bits, labels)
        return loss, loss
    
    (loss, _), grads = jax.value_and_grad(compute_loss_and_grads, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss


@jit
def eval_step(params, model, batch):
    bits, labels = batch
    loss = loss_fn(params, model, bits, labels)
    return loss


def save_checkpoint_to_gcs(bucket_name, checkpoint, checkpoint_path):
    """Save a checkpoint to Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(checkpoint_path)
    
    # Convert JAX arrays in checkpoint to numpy arrays for serialization
    checkpoint_np = jax.tree_map(
        lambda x: np.array(x) if isinstance(x, jax.Array) else x,
        checkpoint
    )
    
    # Save to a temporary file first
    temp_path = f"/tmp/{Path(checkpoint_path).name}"
    np.savez(temp_path, **{"checkpoint": checkpoint_np})
    
    # Upload to GCS
    blob.upload_from_filename(temp_path)
    print(f"Saved checkpoint to gs://{bucket_name}/{checkpoint_path}")


def train(model, params, optimizer, opt_state, train_data, test_data, config, seed):
    train_config = config['train']
    n = config['model']['n']
    bucket_name = "deep-parity-training-0"
    checkpoint_dir = f"checkpoints-1layer-{n}"
    
    model_checkpoints = []
    opt_checkpoints = []
    train_loss_data = []
    test_loss_data = []

    # Set up RNG key
    key = jax.random.PRNGKey(seed)

    for step in tqdm.tqdm(range(train_config['num_steps'])):
        # Get batch
        batch_size = train_config['batch_size']
        train_batch = get_batch(train_data, batch_size, step)
        
        # Train step
        params, opt_state, train_loss = train_step(params, opt_state, model, train_batch)
        
        msg = {'loss/train': float(train_loss)}

        # Eval step
        test_batch = get_batch(test_data, batch_size, step)
        test_loss = eval_step(params, model, test_batch)
        msg['loss/test'] = float(test_loss)
        
        # Fourier analysis (occasionally)
        # if step % 100_000 == 0:
        #     linear_data = fourier_analysis(model, params, n, step)
        #     msg.update(linear_data)
        
        # Save checkpoint
        if step % 500 == 0:
            train_loss_data.append(float(train_loss))
            test_loss_data.append(float(test_loss))
            
            # Create checkpoint
            checkpoint = {
                "params": params,
                "opt_state": opt_state,
                "config": config['model'],
                "rng": np.array(key),  # JAX PRNGKey needs to be converted to numpy
                "step": step
            }
            
            # Save checkpoint to GCS bucket
            checkpoint_path = f"{checkpoint_dir}/{step}.npz"
            save_checkpoint_to_gcs(bucket_name, checkpoint, checkpoint_path)
            
            # Keep checkpoint in memory
            model_checkpoints.append(copy.deepcopy(params))
            opt_checkpoints.append(copy.deepcopy(opt_state))
        
        # Update RNG key
        key, _ = jax.random.split(key)
        
        wandb.log(msg)

    # Save final model
    final_checkpoint = {
        "params": params,
        "config": config['model'],
        "checkpoints": model_checkpoints,
    }
    
    save_checkpoint_to_gcs(bucket_name, final_checkpoint, f"{checkpoint_dir}/full_run.npz")


def main():
    ###########################
    # Configs
    ###########################
    n = 18
    batch_size = 2 ** 16
    frac_train = 0.95
    model_dim = 2048
    optimizer_params = {
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "b1": 0.9,
        "b2": 0.98
    }
    num_steps = 50_000
    seed = 0
    #############################

    # Set JAX seed
    key = jax.random.PRNGKey(seed)
    
    # Prepare dataset
    train_data, test_data = get_datasets(n, frac_train, seed)
    
    # Initialize model
    init_key, key = jax.random.split(key)
    model = Perceptron(n, model_dim)
    params = model.init(init_key)
    
    # Initialize optimizer
    optimizer = optax.adamw(
        learning_rate=optimizer_params["learning_rate"],
        weight_decay=optimizer_params["weight_decay"],
        b1=optimizer_params["b1"],
        b2=optimizer_params["b2"]
    )
    opt_state = optimizer.init(params)
    
    config = {
        "model": {
            "n": n,
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
        group="parity-1Layer-jax",
        project="deep-parity",
        config=config
    )
    
    try:
        train(
            model,
            params,
            optimizer,
            opt_state,
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