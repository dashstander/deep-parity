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


@torch.compile
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
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)
