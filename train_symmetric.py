from argparse import ArgumentParser
from functools import partial
from itertools import product
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import equinox as eqx
import math
import numpy as np
import optax
from pathlib import Path
import polars as pl
import tqdm.auto as tqdm
import wandb
from google.cloud import storage


from algebraist.permutations import Permutation
from algebraist.fourier import sn_fft, sn_ifft
from deep_parity.jax.model import SnPerceptron


parser = ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--n', type=int, default=6, help='Total number of bits')


def make_sn_dataset(n):
    Sn = Permutation.full_group(n)
    indices = list(product(range(len(Sn)), repeat=2))
    targets = [(Sn[i] * Sn[j]).permutation_index() for i, j in indices]
    X = jnp.array(indices)
    return X[:, 0], X[:, 1], jnp.array(targets)


def create_optimizer(config):
    """Create optimizer with learning rate schedule"""
    optimizer_params = config['optim']
    
    # Combine optimizers
    optimizer = optax.adamw(
            learning_rate=optimizer_params["learning_rate"],
            weight_decay=optimizer_params["weight_decay"],
            b1=optimizer_params["b1"],
            b2=optimizer_params["b2"]
        )
    
    
    return optimizer


def make_batch_iterator(X_left, X_right, y, batch_size, sharding):
    """Create batches suitable for TPU training"""
    dataset_size = len(X_right)
    steps_per_epoch = dataset_size // batch_size

    if dataset_size <= batch_size:
        def dumb_iterator(_):
            while True:
                yield jax.device_put((X_left, X_right), sharding), jax.device_put(y, sharding)
        
        return dumb_iterator
    else:
        def data_iterator(key):
            while True:
                key, subkey = jax.random.split(key)
                perm = jax.random.permutation(subkey, dataset_size)
                for i in range(steps_per_epoch):
                    batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                    batch_x = (X_left[batch_idx], X_right[batch_idx])
                    batch_y = y[batch_idx]
                    yield jax.device_put(batch_x, sharding), jax.device_put(batch_y, sharding)
                
        return data_iterator


@partial(jax.jit, static_argnums=4)
def compute_loss(model, x_left, x_right, y, n):
    pred = model(x_left, x_right)
    targets_one_hot = jax.nn.one_hot(
        y,
        num_classes=math.factorial(n)
    )
    loss = optax.softmax_cross_entropy(
        pred,
        targets_one_hot
    )
    # Compute entropy of predictions
    return jnp.mean(loss)


@partial(jax.jit, static_argnums=(1, 5))
def train_step(model, optimizer, opt_state, batch_x, batch_y, n):
    """Training step with gradient update (pmap for TPU parallelism)"""
    x_left, x_right = batch_x
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x_left, x_right, batch_y, n)
    
    # Average gradients across devices
    #grads = jax.lax.pmean(grads, axis_name='batch')
    #loss = jax.lax.pmean(loss, axis_name='batch')
    
    # Apply updates
    updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    new_model = eqx.apply_updates(model, updates)
    
    return new_model, new_opt_state, loss


def save_checkpoint_to_gcs(bucket_name, model, opt_state, rng_key, current_step, config):
    """Save a checkpoint to Google Cloud Storage bucket"""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    # Find checkpoints
    n = config['model']['n']
    model_dim = config['model']['model_dim']
    embed_dim = config['model']['embed_dim']
    seed = config['seed']
    checkpoint_dir = Path(f"symmetric/num_layers=1/n={n}/embed_dim={embed_dim}/model_dim={model_dim}/seed={seed}")
    
    # Save model
    model_local_path = f"/tmp/model_{current_step}.eqx"
    eqx.tree_serialise_leaves(model_local_path, model)
    model_blob = bucket.blob(str(checkpoint_dir / f"model_{current_step}.eqx"))
    model_blob.upload_from_filename(model_local_path)
    
    # Save optimizer state
    opt_local_path = f"/tmp/opt_{current_step}.eqx"
    eqx.tree_serialise_leaves(opt_local_path, opt_state)
    
    opt_blob = bucket.blob(str(checkpoint_dir / f"opt_{current_step}.eqx"))
    opt_blob.upload_from_filename(opt_local_path)
    
    # Save RNG key
    rng_local_path = f"/tmp/rng_{current_step}.npy"
    with open(rng_local_path, "wb") as f:
        np.save(f, jax.device_get(rng_key))
    
    rng_blob = bucket.blob(str(checkpoint_dir / f"rng_{current_step}.npy"))
    rng_blob.upload_from_filename(rng_local_path)
    
    # Save metadata
    meta_local_path = f"/tmp/meta_{current_step}.npy"
    meta = {"step": current_step, "config": config}
    with open(meta_local_path, "wb") as f:
        np.save(f, meta)
    
    meta_blob = bucket.blob(str(checkpoint_dir / f"meta_{current_step}.npy"))
    meta_blob.upload_from_filename(meta_local_path)
    
    print(f"Saved checkpoint to gs://{bucket_name}/{checkpoint_dir} at step {current_step}")


def try_load_checkpoint(model_template, optimizer, bucket_name, config):
    """Try to load the latest checkpoint from GCS bucket"""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    # Find checkpoints
    n = config['model']['n']
    checkpoint_dir = f"checkpoints-1layer-{n}"
    blobs = list(bucket.list_blobs(prefix=f"{checkpoint_dir}/model_"))
    
    if not blobs:
        print("No checkpoints found, starting fresh training")
        return None
    
    # Extract step numbers and find the latest
    steps = []
    for blob in blobs:
        filename = blob.name.split('/')[-1]
        if filename.startswith('model_') and filename.endswith('.eqx'):
            step = int(filename[6:-4])  # Extract step number
            steps.append(step)
    
    if not steps:
        return None
    
    latest_step = max(steps)
    print(f"Found checkpoint at step {latest_step}")
    
    # Download latest checkpoint files
    model_blob = bucket.blob(f"{checkpoint_dir}/model_{latest_step}.eqx")
    model_local_path = f"/tmp/model_{latest_step}.eqx"
    model_blob.download_to_filename(model_local_path)
    
    opt_blob = bucket.blob(f"{checkpoint_dir}/opt_{latest_step}.eqx")
    opt_local_path = f"/tmp/opt_{latest_step}.eqx"
    opt_blob.download_to_filename(opt_local_path)
    
    rng_blob = bucket.blob(f"{checkpoint_dir}/rng_{latest_step}.eqx")
    rng_local_path = f"/tmp/rng_{latest_step}.npy"
    try:
        rng_blob.download_to_filename(rng_local_path)
    except:
        # If RNG file doesn't exist, create a new key
        print("RNG key not found, creating new one")
        rng_key = jax.random.PRNGKey(config['seed'])
    else:
        with open(rng_local_path, "rb") as f:
            rng_key = np.load(f)
    
    # Deserialize model and optimizer state
    model = eqx.tree_deserialise_leaves(model_local_path, model_template)
    
    # Initialize optimizer state with the model template to get the structure
    dummy_opt_state = optimizer.init(eqx.filter(model_template, eqx.is_inexact_array))
    opt_state = eqx.tree_deserialise_leaves(opt_local_path, dummy_opt_state)
    
    return model, opt_state, rng_key, latest_step


def train(config):
    """Main training function"""
    n = config['model']['n']
    embed_dim = config['model']['embed_dim']
    model_dim = config['model']['model_dim']
    batch_size = config['train']['batch_size']
    num_steps = config['train']['num_steps']
    bucket_name = "deep-sn-training-0"
    
    # Set up RNG key
    seed = config['seed']
    key = jax.random.PRNGKey(seed)

    checkpoint_steps = list(range(0, 1000, 20)) + list(range(1000, num_steps + 1, 1000))

    
    # Count devices for data parallelism
    n_devices = jax.device_count()
    per_device_batch = batch_size // n_devices
    print(f"Using {n_devices} devices with {per_device_batch} examples per device")
    mesh = jax.make_mesh((n_devices,), ('batch',))
    sharded = jax.sharding.NamedSharding(mesh, P('batch',))
    replicated = jax.sharding.NamedSharding(mesh, P())

    
    # Generate dataset
    print("Generating dataset...")
    key, data_key = jax.random.split(key)
    
    # Generate all permutations of Sn and their products
    X_left, X_right, y = make_sn_dataset(n)
    
    
    # Shuffle and split data
    num_examples = math.factorial(n) ** 2
    indices = jax.random.permutation(data_key, num_examples)
    split_idx = int(config['train']['frac_train'] * num_examples)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train_left, X_train_right, y_train = X_left[train_indices], X_right[train_indices], y[train_indices]
    X_test_left, X_test_right, y_test =  X_left[test_indices], X_right[test_indices], y[test_indices]
    
    print(f"Train set: {X_train_left.shape[0]} examples")
    print(f"Test set: {X_test_left.shape[0]} examples")
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = create_optimizer(config)
    
    # Initialize model
    print("Initializing model...")
    key, model_key = jax.random.split(key)
    model = SnPerceptron(n, embed_dim, model_dim, model_key)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    model = eqx.filter_shard(model, replicated)
    opt_state = eqx.filter_shard(opt_state, replicated)  

    # Try to load checkpoint
   #checkpoint = try_load_checkpoint(model_template, optimizer, bucket_name, config)

    # Initialize new model and optimizer state
    
    current_step = 0
    
    # Create data iterators
    print("Creating data iterators...")
    key, train_key = jax.random.split(key)
    train_iterator = make_batch_iterator(
        X_train_left, X_train_right, y_train, batch_size, sharded
    )
    
    train_iter = train_iterator(train_key)
    
    # Training loop
    print(f"Starting training for {num_steps} steps...")
    train_losses = []
    test_losses = []
    
    for step in tqdm.tqdm(range(current_step, num_steps)):
        # Get batch
        train_x, train_y = next(train_iter)
        
        # Train step
        #print('About to take a step')
        model, opt_state, train_loss = train_step(model, optimizer, opt_state, train_x, train_y, n)
        train_loss_value = jnp.mean(train_loss).item()
        train_losses.append(train_loss_value)
        #print('Took a single step')
        
        # Log training progress
        msg = {'loss/train': train_loss_value, 'step': step}
        
        # Evaluate on test set periodically
        if step % 100 == 0 and step > 0:
            
            #print(f'Doing eval')
            test_loss = compute_loss(
                model,
                jax.device_put(X_test_left),
                jax.device_put(X_test_right),
                jax.device_put(y_test),
                n
            )
            test_loss_value = jnp.mean(test_loss).item()
            #print(f'Did eval')
            test_losses.append(test_loss_value)
            msg['loss/test'] = test_loss_value
            
            # Fourier analysis
            # Uncomment to enable Fourier analysis (can be slow)
            #if step % 1_000 == 0:
            #    unreplicated_model = jax.device_get(jax.tree.map(lambda x: x[0], model))
            #    fourier_data = fourier_analysis(unreplicated_model, step)
            #    for degree, values in fourier_data.items():
            #        msg[f"fourier/{degree}"] = values
        
        wandb.log(msg)
        #print(f'did logging')
        # Save checkpoint
        if step in checkpoint_steps:
            #print('doing checkpointing')
            save_checkpoint_to_gcs(bucket_name, model, opt_state, key, step, config)
    
    # Save final model
    save_checkpoint_to_gcs(bucket_name, model, opt_state, key, num_steps, config)
    print("Training completed!")


def main(args):
    ###########################
    # Configurations
    ###########################
    seed = args.seed
    n = args.n
    batch_size = 2 ** 19
    frac_train = 0.90
    embed_dim = 128
    model_dim = 512
    optimizer_params = {
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "b1": 0.9,
        "b2": 0.98,
        "max_grad_norm": 1.0
    }
    num_steps = 10_000
    warmup_steps = 0  # Set to 0 to disable warmup
    decay_steps = num_steps - warmup_steps
    ###########################
    
    config = {
        "model": {
            "n": n,
            "model_dim": model_dim,
            "embed_dim": embed_dim
        },
        "optim": optimizer_params,
        "train": {
            "batch_size": batch_size,
            "frac_train": frac_train,
            "num_steps": num_steps,
            "warmup_steps": warmup_steps,
            "decay_steps": decay_steps
        },
        "seed": seed
    }
    
    wandb.init(
        entity='dstander',
        project="symmetric-group-s6-jax",
        config=config
    )
    
    try:
        train(config)
    except KeyboardInterrupt:
        print("Training interrupted")
    
    wandb.finish()


if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    main(args)
