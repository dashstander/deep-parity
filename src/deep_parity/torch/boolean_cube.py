import numpy as np
import torch


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


def generate_boolean_cube(n: int):
    return np.sign(-1. * (generate_all_binary_arrays(n) - 0.5).astype(float))


def get_subcube(n, indices, values):
    assert len(indices) == len(values) and len(indices) < n
    full_cube = torch.from_numpy(generate_all_binary_arrays(n)).to(torch.float32)
    subcube_indices = torch.argwhere(torch.stack([
        full_cube[:, idx] == v for idx, v in zip(indices, values)
    ]).all(dim=0)).squeeze()
    return full_cube[subcube_indices, :]


def fourier_transform(u, normalize=True):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    _, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for _ in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**m if normalize else x.squeeze(-2)
