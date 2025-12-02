import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from gherkin.utils import gll, lagrange1st
from matplotlib import pyplot as plt
from scipy.linalg import lu_factor, lu_solve
import scipy
import time

from mpi4py import MPI
import argparse

rank = MPI.COMM_WORLD.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument('--ncfls', type=int, help='Number of cfl numbers')
parser.add_argument('--nk', type=int, help='Number of wavenumbers')
parser.add_argument('--ncpus', type=int, help='Number of procs', default=1)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

ncfls = args.ncfls
nk = args.nk
ncpus = args.ncpus
run = (not args.plot)
plot = args.plot

data_dir = 'data/stability_1D'
plot_dir = f'plots'
# plot_dir = '../../../latex/ADER Transport/plots'

if rank == 0:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)


def get_matrices(order):
    n = order + 1
    to_st_first = np.zeros((n * n, n))
    for i in range(n):
        to_st_first[i, i] = 1.0

    to_st_last = np.zeros((n * n, n))
    for i in range(n):
        to_st_last[-(n - i), i] = 1.0

    from_st_first = np.zeros((n, n * n))
    for i in range(n):
        from_st_first[i, i] = 1.0

    from_st_last = np.zeros((n, n * n))
    for i in range(n):
        from_st_last[i, -(n - i)] = 1.0

    I = np.eye(n)

    assert np.allclose(from_st_first @ to_st_first, I)
    assert np.allclose(from_st_last @ to_st_last, I)

    xs, w_x = gll(order, iterative=True)
    D = lagrange1st(order, xs).transpose()

    Dx = np.zeros((n, n, n, n))
    Dt = np.zeros((n, n, n, n))

    for i in range(n):
        for j in range(n):
            for k in range(n):
                val = D[i, j]
                Dx[k, i, k, j] = val
                Dt[i, k, j, k] = val

    assert np.allclose(Dt.swapaxes(0, 1).swapaxes(2, 3), Dx)
    Dx = Dx.reshape((D.size, D.size))
    Dt = Dt.reshape(Dx.shape)

    assert np.allclose(from_st_first @ Dx @ to_st_first, D)
    assert np.allclose(from_st_last @ Dx @ to_st_last, D)

    volume_integral = np.zeros((n, n, n, n))

    left_time_integral = np.zeros((n, n, n, n))  # dt when x = -1
    right_time_integral = np.zeros((n, n, n, n))  # dt when x = 1

    first_space_integral = np.zeros((n, n, n, n))  # dx when t = -1
    last_space_integral = np.zeros((n, n, n, n))  # dx when t = 1

    pick_x0_t1 = np.zeros((n, n, n, n))
    pick_x0_t1[:, 0, -1, 0] = 1.0

    right_to_left = np.zeros((n, n, n, n))
    last_to_first = np.zeros((n, n, n, n))

    for i in range(n):
        for j in range(n):
            volume_integral[i, j, i, j] = w_x[i] * w_x[j]

        left_time_integral[i, 0, i, 0] = w_x[i]
        right_time_integral[i, -1, i, -1] = w_x[i]

        first_space_integral[0, i, 0, i] = w_x[i]
        last_space_integral[-1, i, -1, i] = w_x[i]

        right_to_left[i, 0, i, -1] = 1.0
        last_to_first[0, i, -1, i] = 1.0

    volume_integral = volume_integral.reshape(Dx.shape)

    left_time_integral = left_time_integral.reshape(Dx.shape)
    right_time_integral = right_time_integral.reshape(Dx.shape)

    first_space_integral = first_space_integral.reshape(Dx.shape)
    last_space_integral = last_space_integral.reshape(Dx.shape)

    pick_x0_t1 = pick_x0_t1.reshape(Dx.shape)

    right_to_left = right_to_left.reshape(Dx.shape)
    last_to_first = last_to_first.reshape(Dx.shape)

    assert np.allclose(to_st_first @ from_st_last, last_to_first)

    assert np.allclose(left_time_integral @ right_to_left, right_to_left @ right_time_integral)
    assert np.allclose(first_space_integral @ last_to_first, last_to_first @ last_space_integral)

    return (to_st_first, to_st_last, from_st_first, from_st_last, Dx, Dt,
            volume_integral, left_time_integral, right_time_integral, first_space_integral, last_space_integral,
            pick_x0_t1, right_to_left, last_to_first
            )


orders = [3, 5, 7, 9, 11]
top_mag_list = []

if run:

    assert ncpus == MPI.COMM_WORLD.Get_size()

    assert ncfls % ncpus == 0

    cfls = np.linspace(0.0, 1.05, ncfls)

    num_cfl_per_cpu = int(ncfls / ncpus)
    cfls = cfls[rank * num_cfl_per_cpu:(rank + 1) * num_cfl_per_cpu]

    print(f'Rank {rank} cfl range: [{cfls.min()}, {cfls.max()}]')

    for poly_order in orders:
        (to_st_first, to_st_last, from_st_first, from_st_last, Dx, Dt,
         volume_integral, left_time_integral, right_time_integral, first_space_integral, last_space_integral,
         pick_x0_t1, right_to_left, last_to_first
         ) = get_matrices(poly_order)

        inv_vol = np.diag(1 / np.diag(volume_integral))

        left_time_integral = inv_vol @ left_time_integral
        right_time_integral = inv_vol @ right_time_integral
        first_space_integral = inv_vol @ first_space_integral
        last_space_integral = inv_vol @ last_space_integral

        volume_integral = np.eye(volume_integral.shape[0])

        top_mags = []
        for cfl in cfls:
            space_shifts = np.linspace(0, 2 * np.pi, 2000) * 1.0j
            mags = []

            M1 = volume_integral @ (Dt + cfl * Dx) + first_space_integral

            M = volume_integral @ (Dt + cfl * Dx) + first_space_integral + cfl * (left_time_integral - right_time_integral)

            R1 = first_space_integral @ last_to_first @ to_st_last
            R2 = cfl * left_time_integral @ right_to_left  # * np.exp(-space_shifts)
            R4 = cfl * right_time_integral

            pred = scipy.linalg.solve(M1, R1)
            mat1 = from_st_last @ scipy.linalg.solve(M, (R1 - R4 @ pred))
            mat2 = from_st_last @ scipy.linalg.solve(M, R2 @ pred)

            mat = mat1[None] + mat2[None] * np.exp(-space_shifts)[:, None, None]

            eigs = np.linalg.eigvals(mat)
            top_mags.append(abs(eigs).max())

        top_mags = np.array(top_mags)

        suffix = f'order_{poly_order}_rank_{rank}_of_{ncpus}_ncfls_{ncfls}_nk_{nk}.npy'
        fp = os.path.join(data_dir, f'top_mags_{suffix}')
        np.save(fp, top_mags)

        fp = os.path.join(data_dir, f'cfls_{suffix}')
        np.save(fp, cfls)

if plot:

    for poly_order in orders:

        top_mags = []
        cfls = []

        for rank in range(ncpus):
            suffix = f'order_{poly_order}_rank_{rank}_of_{ncpus}_ncfls_{ncfls}_nk_{nk}.npy'
            fp = os.path.join(data_dir, f'top_mags_{suffix}')
            top_mags.extend(np.load(fp))

            fp = os.path.join(data_dir, f'cfls_{suffix}')
            cfls.extend(np.load(fp))

        plt.plot(cfls, np.array(top_mags) - 1, label=f'Order {poly_order}')

    # plt.semilogy(cfls, np.ones_like(top_mags) + 1e-6, '--')
    plt.yscale('symlog', linthresh=1e-14)
    plt.ylabel("Amplification factor")
    plt.xlabel("CFL")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "new-ader-dg-amplification-minus-one.png"))
    plt.show()