import numpy as np
import copy
import scipy.linalg as sci_lin
import kron as kron_ops
import itertools

"""
Integral transforms: Hadamard, Fourier
"""


def apply_hadamard(vecs):
    """
    Applies it our quick way
    Ignores normalization. Assumes no entanglement
    To do it with entanglement, see kron.apply_linear_op_entangled
    """
    return [list(np.array(vec).dot(sci_lin.hadamard(2))) for vec in vecs]


def omega_m(m):
    return np.exp(2 * np.pi * 1j / (2 ** m))


def complex_had(inp):
    return inp.dot(sci_lin.hadamard(2)).astype(np.complex128)


def apply_fourier(vecs, interp):
    """
    This is pretty dire numerically.
    
    Obviously to ship and stuff would figure out something in logspace, etc etc
    """
    vecs = [copy.deepcopy(vec).astype(np.complex128) for vec in vecs]
    res = 1.0 + 0.0j
    for vec_idx, vec in enumerate(vecs):
        res *= complex_had(vec)[interp[vec_idx]]
        if interp[vec_idx] == 1:
            for offset, other_vec_idx in enumerate(range(vec_idx + 1, len(vecs))):
                vecs[other_vec_idx][1] *= omega_m(offset + 2)
    return res


def expand_fourier(vecs):
    """
    This expansion executes the measurement-based semiclassical Fourier transform inspired by Griffiths and Niu.
    But it's not any faster than ordinary FFT, because we are dumping out the whole possible state space
    Realistically, you would be using apply_fourier instead, which is a lot faster than ordinary FFT for Fourier sampling, just like quantum FT
    """
    res = np.zeros(2 ** (len(vecs)), dtype=np.complex128)
    for possible_interp in itertools.product(*[[0, 1] for _ in range(len(vecs))]):
        idx = sum([b << i for i, b in enumerate(possible_interp)])
        res[idx] = apply_fourier(
            [copy.deepcopy(vec).astype(np.complex128) for vec in vecs], possible_interp
        )
    return res


if __name__ == "__main__":
    print(
        """
    Testing 1000 random sets of vectors to see if our fourier transform matches normal FFT results...

    Will bomb out if not correct.

    Entangling in QFT ends up happening w/ superpositions.
    Note kron_opy.rand_vec() will hit us w/ superpositions too, so we are still testing them.
    """
    )
    for idx in range(1000):
        if idx % 30 == 0:
            print(idx)
        vecs = [kron_ops.rand_vec() for _ in range(5)]
        ops = ["*" for _ in range(len(vecs) - 1)]
        expf1 = expand_fourier(vecs)

        expanded_vecs = kron_ops.expand(vecs, ops)
        close_enough = np.allclose(
            np.around(expf1, 3),
            np.around(np.fft.ifft(expanded_vecs) * len(expanded_vecs), 3),
        )
        if not close_enough:
            raise Exception(f"Failure: vecs: {vecs}")
