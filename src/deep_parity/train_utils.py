import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from deep_parity.jax.boolean_cube import generate_boolean_cube, fourier_transform

def make_batch_iterator(X, y, batch_size, n_devices):
    """Create batches suitable for TPU training"""
    dataset_size = len(X)
    steps_per_epoch = max(1, dataset_size // batch_size)
    per_device_batch = batch_size // n_devices
    
    def data_iterator(key):
        while True:
            key, subkey = jax.random.split(key)
            perm = jax.random.permutation(subkey, dataset_size)
            for i in range(steps_per_epoch):
                batch_idx = perm[i * batch_size:(i + 1) * batch_size]
                batch_x = X[batch_idx].reshape(n_devices, per_device_batch, -1)
                batch_y = y[batch_idx].reshape(n_devices, per_device_batch)
                yield jnp.array(batch_x), jnp.array(batch_y)
                
    return data_iterator


def make_base_parity_dataframe(n):
    """Create base dataframe for Fourier analysis"""
    all_binary_data = generate_boolean_cube(n)
    all_parities = all_binary_data.prod(axis=1)
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
    """Calculate power contributions for Fourier analysis"""
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
