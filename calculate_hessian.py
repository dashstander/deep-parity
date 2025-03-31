
import equinox as eqx
from functools import partial
from google.cloud import storage
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from pathlib import Path
from tempfile import TemporaryFile 
from tqdm import tqdm

from deep_parity.jax.model import Perceptron
from deep_parity.jax.boolean_cube import generate_boolean_cube


def try_load_checkpoint(model_template, bucket_name, config, step):
    """Try to load the latest checkpoint from GCS bucket"""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    # Find checkpoints
    n = config['model']['n']
    seed = config['seed']
    model_dim = config['model']['model_dim']
    checkpoint_dir = f"full/one-layer/model_dim={model_dim}/n={n}/seed={seed}"
    blobs = list(bucket.list_blobs(prefix=f"{checkpoint_dir}/model_"))
    
    if not blobs:
        print("No checkpoints found, starting fresh training")
        return None
     
    # Download latest checkpoint files
    model_blob = bucket.blob(f"{checkpoint_dir}/model_{step}.eqx")
    model_local_path = f"/tmp/model_{step}.eqx"
    model_blob.download_to_filename(model_local_path)
    
    model = eqx.tree_deserialise_leaves(model_local_path, model_template)
        
    return model


def upload_hessian(tensor, bucket_name, config, step):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    n = config['model']['n']
    seed = config['seed']
    model_dim = config['model']['model_dim']
    checkpoint_dir = Path(f"full/one-layer/model_dim={model_dim}/n={n}/seed={seed}")

    local_path = TemporaryFile()
    np.save(local_path, np.array(tensor))
    model_blob = bucket.blob(str(checkpoint_dir / f"{step}/"))
    model_blob.upload_from_filename(local_path)


@partial(jax.jit, static_argnums=1)
def degree_n_character(weights, unravel_fn, tensor):
    model = unravel_fn(weights)
    pred = model(tensor)
    logits = jax.nn.log_softmax(pred)[:, 1]
    parities = jnp.expand_dims(tensor.prod(axis=-1), -1)
    # Compute entropy of predictions
    return jnp.mean(parities * logits)


@jax.jit
def calculate_hessian(model, cube, parities, n):
    n_devices = jax.device_count()
    mesh = jax.make_mesh((n_devices,), ('tensor',))
    sharded = jax.sharding.NamedSharding(mesh, P('tensor',))
    replicated = jax.sharding.NamedSharding(mesh, P(None,))

    weights, unravel_fn = jax.tree_flatten.ravel_pytree(model)

    weights = jax.device_put(weights, replicated)
    n_params = len(weights)

    full_hessian = jnp.zeros((n_params, n_params))

    for i in range(0, 2**n, 2**15):
        j = i + 2**15
        hessian_batch = jax.hessian(degree_n_character)(
            weights,
            unravel_fn,
            jax.device_put(cube[i:j], sharded),
            jax.device_put(parities[i:j], sharded)
        )
        full_hessian += hessian_batch
    
    return full_hessian



def main():
    n = 20
    model_dim = 128
    seed = 0
    config = {'model': {'n': n, 'model_dim': model_dim}, 'seed': seed}
    model_bucket = "deep-parity-training-0"
    hessian_bucket = 'deep-parity-hessian'

    key = jax.random.key(0)
    template = Perceptron(n, model_dim, key)

    steps = list(range(0, 2000, 20)) + list(range(2000, 30_000, 1000))
    cube = generate_boolean_cube(n)
    parities = cube.prod(axis=-1)

    for step in tqdm(steps):

        model = try_load_checkpoint(template, model_bucket, config, step)
        hessian = calculate_hessian(model, cube, parities, n)
        upload_hessian(hessian, hessian_bucket, config, step)

    
if __name__ == '__main__':
    main()
