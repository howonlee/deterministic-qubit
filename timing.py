import numpy as np
import kron as kron_ops
import int_trans
import random
import matplotlib.pyplot as plt
import scipy.linalg as sci_lin
from timeit import default_timer as perf_timer

MEDIAN_OF = 5


def fwht(a):
    """ Inplace, taken from Wikipedia
    https://en.wikipedia.org/wiki/Fast_Walsh%E2%80%93Hadamard_transform
    """
    h = 1
    while h < len(a):
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + y
                a[j + h] = x - y
        h *= 2


def plot_entangled_op():
    """
    Linear op is Hadamard product
    Not very sporting to actually instantiate full Hadamard product w/o fast impl,
    so a reasonably fast one is up above.
    """
    full_exp_res = []
    prop_res = []
    var_range = list(range(2, 20))
    for var_size in var_range:
        print(var_size)
        curr_exp_res = []
        curr_prop_res = []
        for curr_idx in range(MEDIAN_OF):
            curr_vecs = [kron_ops.rand_vec() for _ in range(var_size)]
            curr_ops = [kron_ops.rand_op() for _ in range(var_size - 1)]
            linear_op_mats = [sci_lin.hadamard(2) for _ in range(var_size)]

            full_start = perf_timer()
            full_expansion = kron_ops.expand(curr_vecs, curr_ops)
            fwht(full_expansion)
            full_end = perf_timer()
            full_exp_time = full_end - full_start

            prop_start = perf_timer()
            our_way = kron_ops.apply_linear_op_entangled(
                linear_op_mats, curr_vecs, curr_ops
            )
            prop_end = perf_timer()
            prop_res_time = prop_end - prop_start

            curr_exp_res.append(full_exp_time)
            curr_prop_res.append(prop_res_time)
        full_exp_res.append(sorted(curr_exp_res)[MEDIAN_OF // 2])
        prop_res.append(sorted(curr_prop_res)[MEDIAN_OF // 2])

    xscale = [2 ** var_size for var_size in var_range]
    (full,) = plt.plot(xscale, full_exp_res, color="blue", label="Full expansion")
    (prop,) = plt.plot(xscale, prop_res, color="red", label="Propositional method")
    plt.legend(handles=[full, prop])
    plt.title("Time to left mult a factorable operator and random entangled state")
    plt.xlabel("Number data points")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("entangling.png")
    plt.clf()


def plot_qft():
    full_exp_res = []
    prop_res = []
    var_range = list(range(2, 20))
    for var_size in var_range:
        print(var_size)
        curr_exp_res = []
        curr_prop_res = []
        for curr_idx in range(MEDIAN_OF):
            curr_vecs = [kron_ops.rand_vec() for _ in range(var_size)]
            curr_ops = ["*" for _ in range(var_size)]
            curr_interp = [random.choice([0, 1]) for _ in range(var_size)]

            full_start = perf_timer()
            full_expansion = kron_ops.expand(curr_vecs, curr_ops)
            np.fft.ifft(full_expansion)
            full_end = perf_timer()
            full_exp_time = full_end - full_start

            prop_start = perf_timer()
            our_way = int_trans.apply_fourier(curr_vecs, curr_interp)
            prop_end = perf_timer()
            prop_res_time = prop_end - prop_start

            curr_exp_res.append(full_exp_time)
            curr_prop_res.append(prop_res_time)
        full_exp_res.append(sorted(curr_exp_res)[MEDIAN_OF // 2])
        prop_res.append(sorted(curr_prop_res)[MEDIAN_OF // 2])

    xscale = [2 ** var_size for var_size in var_range]
    (full,) = plt.plot(xscale, full_exp_res, color="blue", label="FFT")
    (prop,) = plt.plot(xscale, prop_res, color="red", label="Propositional FT")
    plt.legend(handles=[full, prop])
    plt.title("Ordinary FFT  vs. propositional Fourier transform, time")
    plt.xlabel("Number data points")
    plt.ylabel("Time (s)")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig("qft.png")
    plt.clf()


if __name__ == "__main__":
    plot_entangled_op()
    plot_qft()
