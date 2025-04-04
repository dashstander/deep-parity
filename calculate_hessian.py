
from argparse import ArgumentParser
import equinox as eqx
from functools import partial
from google.cloud import storage
import jax
import jax.flatten_util
from jax.nn import relu
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
from pathlib import Path
from tqdm import tqdm

from deep_parity.jax.model import Perceptron
from deep_parity.jax.boolean_cube import generate_boolean_cube


parser = ArgumentParser()
parser.add_argument('--seed', type=int)
parser.add_argument('--n', type=int)
parser.add_argument('--model_dim', type=int)


class Perceptron(eqx.Module):
    linear: eqx.Module
    unembed: eqx.Module

    def __init__(self, n: int, model_dim: int, key, use_bias=True):
        linear_key, unembed_key = jax.random.split(key)
        self.linear = eqx.nn.Linear(in_features=n, out_features=model_dim, use_bias=use_bias, key=linear_key)
        self.unembed = eqx.nn.Linear(in_features=model_dim, out_features=2, use_bias=False, key=unembed_key)
    
    def __call__(self, x):
        preactivations = self.linear(x)
        outputs = self.unembed(relu(preactivations))
        return outputs



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
    checkpoint_dir = Path(f"fisher_information_matrix/one-layer/model_dim={model_dim}/n={n}/seed={seed}")

    local_path = f"/tmp/{step}.npy"
    np.save(local_path, tensor)
    model_blob = bucket.blob(str(checkpoint_dir / f"{step}/fim{step}.npy"))
    model_blob.upload_from_filename(local_path)


def model_logits(model, tensor):
    logits = jax.nn.log_softmax(model(tensor))
    return logits


def softmax_jacobian(probs):
    return jnp.diag(probs) - jnp.outer(probs, probs) 


@partial(jax.vmap, in_axes=(None, 0))
def softmax_gradient_outer_product(model, tensor):
    n_classes = 2
    
    def grad_fn(idx):
        vals, grads = jax.value_and_grad(lambda m, x: model_logits(m, x)[idx])(model, tensor)
        flat_grads, _ = jax.flatten_util.ravel_pytree(grads)
        return vals, flat_grads
    
    logits, gradients = jax.vmap(grad_fn)(jnp.arange(n_classes))
    # I want it to have shape (n_params, n_classes)
    gradients = gradients.T

    # Amari calls it `Q` for some reason
    Q = softmax_jacobian(jax.nn.softmax(logits))
    return gradients @ Q @ gradients.T


@jax.jit
def fisher_information(model, tensor):
    return softmax_gradient_outer_product(model, tensor).mean(axis=0)


def calculate_fim(model, cube, n):
    n_devices = jax.device_count()
    mesh = jax.make_mesh((n_devices,), ('tensor',))
    sharded = jax.sharding.NamedSharding(mesh, P('tensor',))
    replicated = jax.sharding.NamedSharding(mesh, P(None,))
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(model))

    full_fisher_information = np.zeros((n_params, n_params), dtype=np.float64)

    model = jax.device_put(model, replicated)

    for i in range(0, 2**n, 2**15):
        j = i + 2**15
        fim_batch = fisher_information(
            model,
            jax.device_put(cube[i:j], sharded),
        )
        full_fisher_information += np.array(fim_batch, dtype=np.float64)
    
    return full_fisher_information


def main(args):
    n = args.n
    model_dim = args.model_dim
    seed = args.seed
    config = {'model': {'n': n, 'model_dim': model_dim}, 'seed': seed}
    model_bucket = "deep-parity-training-0"
    hessian_bucket = 'deep-parity-hessian'

    key = jax.random.key(0)
    template = Perceptron(n, model_dim, key)

    steps = list(range(0, 2000, 20)) + list(range(2000, 30_001, 1000))
    cube = generate_boolean_cube(n)

    for step in tqdm(steps):

        model = try_load_checkpoint(template, model_bucket, config, step)
        hessian = calculate_fim(model, cube, n)
        upload_hessian(hessian, hessian_bucket, config, step)

    
if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    main(args)
