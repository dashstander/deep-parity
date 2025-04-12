from argparse import ArgumentParser
import equinox as eqx
from google.cloud import storage
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P
import numpy as np
import optax
from pathlib import Path
import polars as pl
import scipy
from tqdm import tqdm

from deep_parity.jax.boolean_cube import fourier_transform, generate_boolean_cube
from deep_parity.jax.model import Perceptron


parser = ArgumentParser()
parser.add_argument('--n', type=int)
parser.add_argument('--seed', type=int)
parser.add_argument('--model_dim', type=int)
parser.add_argument('--top_k', type=int, default=100)


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
        raise ValueError('No models found.')
    
    # Download latest checkpoint files
    model_blob = bucket.blob(f"{checkpoint_dir}/model_{step}.eqx")
    model_local_path = f"/tmp/model_{step}.eqx"
    model_blob.download_to_filename(model_local_path)

  
    model = eqx.tree_deserialise_leaves(model_local_path, model_template)
    
    return model


def try_load_fim(bucket_name, config, step):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    
    n = config['model']['n']
    seed = config['seed']
    model_dim = config['model']['model_dim']
    checkpoint_dir = Path(f"fisher_information_matrix/one-layer/model_dim={model_dim}/n={n}/seed={seed}")
        
    # Download latest checkpoint files
    tensor_blob = bucket.blob(str(f"{checkpoint_dir}/{step}/fim{step}.npy"))
    tensor_blob_path = f"/tmp/{step}.npy"
    tensor_blob.download_to_filename(tensor_blob_path)
    
    return np.load(tensor_blob_path)


def save_and_upload_parquet(df, bucket_name, path, filename, model_metadata=None, partitions=100, overwrite=False):
    """
    Save a Polars DataFrame as a Parquet file and upload it to a GCS bucket.
    
    Args:
        df: Polars DataFrame to save
        bucket_name: GCS bucket name
        path: Path within the bucket
        filename: Name of the file (without .parquet extension)
        model_metadata: Dict containing model metadata to include as columns (n, seed, model_dim)
        partitions: Number of row groups/partitions for the parquet file
        overwrite: Whether to overwrite existing files
    """
    # Create local directory if it doesn't exist
    local_dir = Path("/tmp/parquet_files")
    local_dir.mkdir(exist_ok=True)
    
    # Add model metadata as columns if provided
    if model_metadata:
        for key, value in model_metadata.items():
            df = df.with_columns(pl.lit(value).alias(key))
    
    # Save the DataFrame locally as a Parquet file
    local_path = local_dir / f"{filename}.parquet"
    
    # Configure parquet writing options for efficient querying
    parquet_options = {
        "row_group_size": max(1, len(df) // partitions),  # Ensure at least 1 row per group
        "compression": "snappy",                          # Good balance of compression/speed
        "statistics": True                                # Enable column statistics for filtering
    }
    
    df.write_parquet(local_path, **parquet_options)
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Create the full path for the destination
    destination_blob_name = f"{path}/{filename}.parquet"
    blob = bucket.blob(destination_blob_name)
    
    # Check if file exists and handle overwrite flag
    if blob.exists() and not overwrite:
        print(f"File {destination_blob_name} already exists. Skipping upload.")
        local_path.unlink()
        return
    
    # Upload the file
    blob.upload_from_filename(local_path)
    
    print(f"File {filename}.parquet uploaded to gs://{bucket_name}/{destination_blob_name}")
    
    # Clean up the local file
    local_path.unlink()


def compute_loss(model, batch_x, batch_y):
    logits = model(batch_x)
    
    target_one_hot = jax.nn.one_hot(
        (batch_y == 1.).astype(int),
        num_classes=2
    )
    
    loss = optax.softmax_cross_entropy(
        logits,
        target_one_hot
    )
    
    return loss.mean()


def compute_fourier_statistics(base_fourier, perturbed_fourier, idx=None):
    """
    Compute statistical summaries of the differences between Fourier coefficients
    of the base model and perturbed models.
    
    Args:
        base_fourier: np.array of Fourier coefficients from the base model
        perturbed_fourier: np.array of Fourier coefficients from the perturbed model
        idx: Optional index of the perturbation (e.g., eigenvector index)
        
    Returns:
        Dict of summary statistics
    """
    difference = perturbed_fourier - base_fourier
    
    # Count non-zero elements in bit representation to get Fourier degree
    n_bits = int(np.log2(len(base_fourier)))
    degrees = np.array([bin(i).count('1') for i in range(len(base_fourier))])
    
    # Group differences by degree
    stats = {}
    
    # Overall statistics
    stats["mean_diff"] = float(np.mean(difference))
    stats["std_diff"] = float(np.std(difference))
    stats["l1_norm"] = float(np.sum(np.abs(difference)))
    stats["l2_norm"] = float(np.sqrt(np.sum(difference**2)))
    stats["max_diff_idx"] = int(np.argmax(np.abs(difference)))
    stats["max_diff_value"] = float(difference[stats["max_diff_idx"]])
    stats["max_diff_degree"] = int(degrees[stats["max_diff_idx"]])
    
    # Per-degree statistics
    for d in range(n_bits + 1):
        degree_mask = (degrees == d)
        if np.sum(degree_mask) > 0:
            degree_diff = difference[degree_mask]
            stats[f"degree_{d}_mean"] = float(np.mean(degree_diff))
            stats[f"degree_{d}_std"] = float(np.std(degree_diff))
            stats[f"degree_{d}_l1_norm"] = float(np.sum(np.abs(degree_diff)))
            stats[f"degree_{d}_l2_norm"] = float(np.sqrt(np.sum(degree_diff**2)))
            stats[f"degree_{d}_max_coef_idx"] = int(np.argmax(np.abs(degree_diff)))
            local_max_idx = np.arange(len(base_fourier))[degree_mask][stats[f"degree_{d}_max_coef_idx"]]
            stats[f"degree_{d}_max_coef_global_idx"] = int(local_max_idx)
            stats[f"degree_{d}_max_coef_value"] = float(difference[local_max_idx])
    
    # Add perturbation index if provided
    if idx is not None:
        stats["perturbation_idx"] = idx
        
    return stats


def calculate_fisher_projections(model, fisher_matrix, boolean_cube, parities, step):
    n_devices = jax.device_count()
    mesh = jax.make_mesh((n_devices,), ('tensor',))
    sharded = jax.sharding.NamedSharding(mesh, P('tensor',))
    replicated = jax.sharding.NamedSharding(mesh, P())

    model = jax.device_put(model, replicated)
    
    eigvals, eigvecs = scipy.linalg.eigh(fisher_matrix)
    loss, model_grads = jax.value_and_grad(compute_loss)(
        model, jax.device_put(boolean_cube, sharded),
        jax.device_put(parities, sharded))
    grads, _ = jax.flatten_util.ravel_pytree(model_grads)
    grads /= jnp.sqrt(jnp.pow(grads, 2).sum())
    correlations = np.array(jnp.array(eigvecs).T @ grads)

    data = pl.DataFrame({'eigenvalues': eigvals, 'gradient_projections': correlations}).with_row_index().with_columns(step=pl.lit(step), loss=pl.lit(loss))
    return data, eigvecs


def decompose_gradients(proj_data, model, eigvecs, boolean_cube, step, top_k=100):
    n_devices = jax.device_count()
    mesh = jax.make_mesh((n_devices,), ('tensor',))
    sharded = jax.sharding.NamedSharding(mesh, P('tensor',))
    replicated = jax.sharding.NamedSharding(mesh, P())

    model = jax.device_put(model, replicated)
    boolean_cube = jax.device_put(boolean_cube, sharded) 

    weights, unravel_fn = jax.flatten_util.ravel_pytree(model)

    base_logits = model(boolean_cube)

    base_logits_ft = fourier_transform(base_logits.T).T
    even_logit_fourier = {'base': np.array(base_logits_ft[:, 0])}
    odd_logit_fourier = {'base': np.array(base_logits_ft[:, 1])}

    even_summary_data = []
    odd_summary_data = []

    top_eigenvector_indices = proj_data.sort(pl.col('gradient').abs()).tail(top_k)['index'].to_list()

    for idx in top_eigenvector_indices:
        proj_sign = jnp.sign(proj_data.filter(pl.col('index').eq(idx))['gradient_projections'].item())
        perturbed_model = unravel_fn(weights + (proj_sign * jnp.array(eigvecs[:, idx])))
        perturbed_model = jax.device_put(perturbed_model, replicated)
        perturbed_logits = model(boolean_cube)
        perturbed_logits_ft = fourier_transform(perturbed_logits.T).T
        even_logit_fourier[str(idx)] = np.array(perturbed_logits_ft[:, 0])
        odd_logit_fourier[str(idx)] = np.array(perturbed_logits_ft[:, 1])
        even_summary = compute_fourier_statistics(base_logits_ft[:, 0], perturbed_logits_ft[:, 0], idx)
        odd_summary = compute_fourier_statistics(base_logits_ft[:, 1], perturbed_logits_ft[:, 1], idx)
        odd_summary_data.append(odd_summary)
        even_summary_data.append(even_summary)
    
    even_logit_df = pl.DataFrame(even_logit_fourier).with_columns(step=pl.lit(step)).with_row_index()
    odd_logit_df = pl.DataFrame(odd_logit_fourier).with_columns(step=pl.lit(step)).with_row_index()
    even_summary_df = pl.concat(even_summary_data).with_columns(step=pl.lit(step))
    odd_summary_df = pl.concat(odd_summary_data).with_columns(step=pl.lit(step))

    return even_logit_df, odd_logit_df, even_summary_df, odd_summary_df



def main(args):
    seed = args.seed
    n = args.n
    model_dim = args.model_dim
    config = {'model': {'n': n, 'model_dim': model_dim}, 'seed': seed}

    model_bucket = "deep-parity-training-0"
    hessian_bucket = "deep-parity-hessian"
    output_bucket = 'deep-parity-analysis-0'

    partitions = 100
    overwrite = False

    key = jax.random.key(0)

    template = Perceptron(n, model_dim, key)

    # Create more queryable directory structure
    # Instead of nesting directories by parameters, use a flat structure with metadata in the dataframes
    analysis_type = "one-layer"
    
    # Define paths for different data types - use flat structure for better partitioning
    fisher_proj_path = f"analysis/fisher_projections"
    even_logit_path = f"analysis/even_logit_fourier" 
    odd_logit_path = f"analysis/odd_logit_fourier"
    even_stats_path = f"analysis/even_fourier_stats"
    odd_stats_path = f"analysis/odd_fourier_stats"
    
    # Create model metadata dictionary with all relevant parameters for querying
    model_metadata = {
        "n": n,
        "seed": seed,
        "model_dim": model_dim,
        "analysis_type": analysis_type,
        "architecture": "perceptron"  # Add more metadata that might be useful for filtering
    }

    all_steps = list(range(0, 2000, 20)) + list(range(2000, 30_001, 1000))
    
    cube = jnp.array(generate_boolean_cube(n))
    parities = cube.prod(axis=-1)

    for step in tqdm(all_steps):

        model = try_load_checkpoint(template, model_bucket, config, step)
        FIM = try_load_fim(hessian_bucket, config, step)

        fisher_proj_data, eigenvectors = calculate_fisher_projections(model, FIM, cube, parities, step)
        even_logit_df, odd_logit_df, even_stats_df, odd_stats_df = decompose_gradients(
            fisher_proj_data,
            model,
            eigenvectors,
            cube,
            step,
            args.top_k
        )

        # Add step to metadata for each individual file
        step_metadata = model_metadata.copy()
        step_metadata["step"] = step

        # Save individual step results with model metadata included
        save_and_upload_parquet(
            fisher_proj_data, 
            output_bucket, 
            fisher_proj_path, 
            f"fisher_projections_n={n}_seed={seed}_model_dim={model_dim}_step={step}",
            step_metadata,
            partitions,
            overwrite
        )
            
        save_and_upload_parquet(
            even_logit_df, 
            output_bucket, 
            even_logit_path, 
            f"even_logit_fourier_n={n}_seed={seed}_model_dim={model_dim}_step={step}",
            step_metadata,
            partitions,
            overwrite
        )
            
        save_and_upload_parquet(
            odd_logit_df, 
            output_bucket, 
            odd_logit_path, 
            f"odd_logit_fourier_n={n}_seed={seed}_model_dim={model_dim}_step={step}",
            step_metadata,
            partitions,
            overwrite
        )
        
        # Save Fourier statistics
        save_and_upload_parquet(
            even_stats_df, 
            output_bucket, 
            even_stats_path, 
            f"even_fourier_stats_n={n}_seed={seed}_model_dim={model_dim}_step={step}",
            step_metadata,
            partitions,
            overwrite
        )
            
        save_and_upload_parquet(
            odd_stats_df, 
            output_bucket, 
            odd_stats_path, 
            f"odd_fourier_stats_n={n}_seed={seed}_model_dim={model_dim}_step={step}",
            step_metadata,
            partitions,
            overwrite
        )
        


if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    main(args)