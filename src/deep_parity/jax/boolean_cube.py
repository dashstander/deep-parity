from functools import partial
import numpy as np
import jax
import jax.numpy as jnp


def generate_all_binary_arrays(n: int):
    """Create an array of all possible binary sequences on n bits.
    Parameters:
        n: integer, number of bits in the output
    Returns:
        x: numpy array with shape (2**n - 1, n) and every possible binary sequence.
    """
    # Create an array of all possible numbers for n bits
    numbers = np.arange(2**n, dtype=np.uint32)
    # Use broadcasting with a single bitwise operation
    return ((numbers[:, np.newaxis] >> np.arange(n)[::-1]) & 1).astype(np.uint8)


def get_subcube(n, indices, values):
    assert len(indices) == len(values) and len(indices) < n
    full_cube = jnp.array(generate_all_binary_arrays(n))
    subcube_indices = jnp.argwhere(jnp.stack([
        full_cube[:, idx] == v for idx, v in zip(indices, values)
    ]).all(axis=0)).squeeze()
    return full_cube[subcube_indices, :]


def generate_boolean_cube(n: int):
    return np.sign(-1. * (generate_all_binary_arrays(n) - 0.5).astype(float))


@partial(jax.jit, static_argnums=1)
def fourier_transform(u, normalize=True):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^m where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    _, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]

    def _step(_, tensor):
        return jnp.concat((tensor[..., ::2, :] + tensor[..., 1::2, :], tensor[..., ::2, :] - tensor[..., 1::2, :]), dim=-1)

    fft = jax.lax.fori_loop(0, m, _step, x)
    return fft.squeeze(-2) / 2**m if normalize else x.squeeze(-2)
