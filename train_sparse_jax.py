from argparse import ArgumentParser
from functools import partial
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import optax
from pathlib import Path
import polars as pl
import tqdm.auto as tqdm
import wandb
from google.cloud import storage

from deep_parity.jax.boolean_cube import generate_boolean_cube
from deep_parity.jax.model import Perceptron
from deep_parity.train_utils import calc_power_contributions, make_batch_iterator


parser = ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--n', type=int, default=20, help='Total number of bits')
parser.add_argument('--k', type=int, default=5, help='Number of first k bits to calculate parity on')


def get_activations(model, n):
    """Get activations for all binary arrays of length n"""
    batch_size = 2 ** 16
    bits = generate_boolean_cube(n)
    
    # Split into batches
    num_batches = (bits.shape[0] + batch_size - 1) // batch_size
    
    @jax.jit
    def get_activations_batch(batch):
        # Extract the linear layer activations
        return jax.nn.relu(model(batch))
    
    activations = []
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, bits.shape[0])
        batch = bits[start_idx:end_idx]
        acts = get_activations_batch(batch)
        activations.append(acts)
    
    return jnp.concatenate(activations)


def fourier_analysis(model, epoch):
    """Perform Fourier analysis on the model"""
    linear_preacts = get_activations(model, model.linear.in_features)
    embed_power_df = calc_power_contributions(linear_preacts, model.linear.in_features, epoch)
    return embed_power_df


def create_optimizer(config):
    """Create optimizer with learning rate schedule"""
    optimizer_params = config['optim']
    
    # Linear warmup and decay schedule
    warmup_steps = config['train']['warmup_steps']
    decay_steps = config['train']['decay_steps']
    
    if warmup_steps > 0:
        warmup_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=optimizer_params["learning_rate"],
            transition_steps=warmup_steps
        )
        decay_schedule = optax.linear_schedule(
            init_value=optimizer_params["learning_rate"],
            end_value=0.0,
            transition_steps=decay_steps
        )
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, decay_schedule],
            boundaries=[warmup_steps]
        )
    else:
        schedule = optimizer_params["learning_rate"]
    
    # Combine optimizers
    optimizer = optax.chain(
        optax.clip_by_global_norm(optimizer_params.get("max_grad_norm", 1.0)),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=optimizer_params["weight_decay"],
            b1=optimizer_params["b1"],
            b2=optimizer_params["b2"]
        )
    )
    
    return optimizer


def evaluate_model(model, X_test, y_test, n_devices):
    """Evaluate model on test set ensuring TPU compatibility"""
    # Make sure test data size is divisible by n_devices
    test_size = X_test.shape[0]
    padded_size = ((test_size + n_devices - 1) // n_devices) * n_devices
    
    # Pad the test data if needed
    if test_size < padded_size:
        # Create padding
        pad_size = padded_size - test_size
        pad_X = jnp.zeros((pad_size,) + X_test.shape[1:])
        pad_y = jnp.zeros(pad_size)
        
        # Concatenate padding
        X_padded = jnp.concatenate([X_test, pad_X], axis=0)
        y_padded = jnp.concatenate([y_test, pad_y], axis=0)
    else:
        X_padded = X_test
        y_padded = y_test
    
    # Reshape for PMapped evaluation
    per_device = padded_size // n_devices
    X_reshaped = X_padded.reshape(n_devices, per_device, -1)
    y_reshaped = y_padded.reshape(n_devices, per_device)
    
    # Run evaluation
    padded_loss = eval_step(model, X_reshaped, y_reshaped)
    
    # Average loss (all devices return same value due to pmean in eval_step)
    return jnp.mean(padded_loss)


def compute_loss(model, batch_x, batch_y):
    pred = model(batch_x)
    
    targets_one_hot = jax.nn.one_hot(
        (batch_y == 1.).astype(int),
        num_classes=2
    )
    
    per_example_loss = optax.softmax_cross_entropy(
        pred,
        targets_one_hot
    )
    
    losses = jnp.mean(per_example_loss, axis=0)
    return jnp.mean(losses)


def _train_step(optimizer, model, opt_state, batch_x, batch_y):
    """Training step with gradient update (pmap for TPU parallelism)"""
    loss, grads = eqx.filter_value_and_grad(compute_loss)(model, batch_x, batch_y)
    
    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    
    # Apply updates
    updates, new_opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
    new_model = eqx.apply_updates(model, updates)
    
    return new_model, new_opt_state, loss


@partial(jax.pmap, axis_name='batch')
def eval_step(model, batch_x, batch_y):
    """Evaluation step (pmap for TPU parallelism)"""
    loss = compute_loss(model, batch_x, batch_y)
    return jax.lax.pmean(loss, axis_name='batch')


def save_checkpoint_to_gcs(bucket_name, model, opt_state, rng_key, current_step, config):
    """Save a checkpoint to Google Cloud Storage bucket"""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    # Create checkpoint directory path
    n = config['model']['n']
    k = config['model']['k']
    model_dim = config['model']['model_dim']
    seed = config['seed']
    checkpoint_dir = Path(f"sparse/one-layer/model_dim={model_dim}/n={n}/k={k}/seed={seed}") /f"{current_step}"

    
    # Save model
    model_local_path = f"/tmp/model_{current_step}.eqx"
    eqx.tree_serialise_leaves(model_local_path, jax.device_get(jax.tree.map(lambda x: x[0], model)))
    
    model_blob = bucket.blob(str(checkpoint_dir / f"model_{current_step}.eqx"))
    model_blob.upload_from_filename(model_local_path)
    
    # Save optimizer state
    opt_local_path = f"/tmp/opt_{current_step}.eqx"
    eqx.tree_serialise_leaves(opt_local_path, jax.device_get(jax.tree.map(lambda x: x[0], opt_state)))
    
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
    k = config['model']['k']
    model_dim = config['model']['model_dim']
    seed = config['seed']
    checkpoint_dir = f"sparse/one-layer/model_dim={model_dim}/n={n}/k={k}/seed={seed}"
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


def compute_sparse_parity(sequences, k):
    """Compute parity of the first k bits of each sequence"""
    return sequences[:, :k].prod(axis=1)


def train(config):
    """Main training function"""
    n = config['model']['n']
    k = config['model']['k']
    model_dim = config['model']['model_dim']
    batch_size = config['train']['batch_size']
    num_steps = config['train']['num_steps']
    bucket_name = "deep-parity-training-0"

    checkpoint_steps = list(range(0, 1000, 20)) + list(range(1000, num_steps + 1, 1000))
    
    # Set up RNG key
    seed = config['seed']
    key = jax.random.PRNGKey(seed)
    
    # Count devices for data parallelism
    n_devices = jax.device_count()
    per_device_batch = batch_size // n_devices
    print(f"Using {n_devices} devices with {per_device_batch} examples per device")
    
    # Generate dataset
    print(f"Generating dataset for sparse parity (first {k} of {n} bits)...")
    key, data_key = jax.random.split(key)
    
    # Generate all binary arrays but compute parities only for the first k bits
    sequences = generate_boolean_cube(n)
    parities = compute_sparse_parity(sequences, k)
    
    # Shuffle and split data
    num_examples = sequences.shape[0]
    indices = jax.random.permutation(data_key, num_examples)
    split_idx = int(config['train']['frac_train'] * num_examples)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, y_train = sequences[train_indices], parities[train_indices]
    X_test, y_test = sequences[test_indices], parities[test_indices]
    
    print(f"Train set: {X_train.shape[0]} examples")
    print(f"Test set: {X_test.shape[0]} examples")
    
    # Create optimizer
    print("Creating optimizer...")
    optimizer = create_optimizer(config)
    
    # Initialize model
    print("Initializing model...")
    key, model_key = jax.random.split(key)
    model = Perceptron(n, model_dim, model_key)
    
    # Try to load checkpoint
    #checkpoint = try_load_checkpoint(model, optimizer, bucket_name, config)
    
    # Initialize new model and optimizer state
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    current_step = 0
    
    # Replicate model and optimizer state across devices
    model = jax.device_put_replicated(model, jax.devices())
    opt_state = jax.device_put_replicated(opt_state, jax.devices())
    
    # Create data iterators
    print("Creating data iterators...")
    key, train_key = jax.random.split(key)
    train_iterator = make_batch_iterator(
        X_train, y_train, batch_size, n_devices
    )
    
    train_iter = train_iterator(train_key)
    
    # Training loop
    print(f"Starting training for {num_steps} steps...")
    train_losses = []
    test_losses = []

    train_step = jax.pmap(partial(_train_step, optimizer), axis_name='batch')
    
    for step in tqdm.tqdm(range(current_step, num_steps)):
        # Get batch
        train_x, train_y = next(train_iter)
        
        # Train step
        model, opt_state, train_loss = train_step(model, opt_state, train_x, train_y)
        train_loss_value = jnp.mean(train_loss).item()
        train_losses.append(train_loss_value)
        
        # Log training progress
        msg = {'loss/train': train_loss_value, 'step': step}
        
        # Evaluate on test set periodically
        if step % 100 == 0 and step > 0:
            test_loss = evaluate_model(model, X_test, y_test, n_devices)
            test_loss_value = jnp.mean(test_loss).item()
            test_losses.append(test_loss_value)
            msg['loss/test'] = test_loss_value
            
            # Fourier analysis
            # Uncomment to enable Fourier analysis (can be slow)
            if step % 1_000 == 0:
                unreplicated_model = jax.device_get(jax.tree.map(lambda x: x[0], model))
                fourier_data = fourier_analysis(unreplicated_model, step)
                for degree, values in fourier_data.items():
                    msg[f"fourier/{degree}"] = jnp.mean(values)
        
        wandb.log(msg)
        
        # Save checkpoint
        if step in checkpoint_steps:
            save_checkpoint_to_gcs(bucket_name, model, opt_state, key, step, config)
    
    # Save final model
    save_checkpoint_to_gcs(bucket_name, model, opt_state, key, num_steps, config)
    print("Training completed!")


def main(args):
    ###########################
    # Configurations
    ###########################
    n = args.n if args.n is not None else 20
    k = args.k if args.k is not None else 5
    batch_size = 2 ** 18
    frac_train = 0.90
    model_dim = 2048
    optimizer_params = {
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "b1": 0.9,
        "b2": 0.98,
        "max_grad_norm": 1.0
    }
    num_steps = 50_000
    warmup_steps = 100  # Set to 0 to disable warmup
    decay_steps = num_steps - warmup_steps
    seed = int(args.seed)
    ###########################
    
    config = {
        "model": {
            "n": n,
            "k": k,
            "model_dim": model_dim,
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
        group=f"sparse-parity-1Layer-k{k}",
        project="deep-parity-jax",
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