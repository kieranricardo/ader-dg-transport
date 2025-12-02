import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from ader_dg_transport.utils import gll, lagrange1st
from matplotlib import pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D
import scipy
import time

from mpi4py import MPI
import argparse

rank = MPI.COMM_WORLD.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument('--nk', type=int, help='Number of wave numbers')
parser.add_argument('--ncpus', type=int, help='Number of procs', default=1)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

nk = args.nk
ncpus = args.ncpus
run = (not args.plot)
plot = args.plot

data_dir = 'data/stability_2D'
plot_dir = f'plots'
# plot_dir = '../../../latex/ADER Transport/plots'

if rank == 0:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)


def get_matrices(poly_order):
    n = poly_order + 1

    shape = (n,) * 6

    xp_to_xm = np.zeros(shape)
    yp_to_ym = np.zeros(shape)

    for i in range(n):
        for j in range(n):
            xp_to_xm[i, 0, j, i, -1, j] = 1.0
            yp_to_ym[i, j, 0, i, j, -1] = 1.0

    xp_to_xm = xp_to_xm.reshape((n ** 3, n ** 3))
    yp_to_ym = yp_to_ym.reshape((n ** 3, n ** 3))

    shape = (n,) * 5
    to_st_first = np.zeros(shape)
    to_st_last = np.zeros(shape)

    from_st_first = np.zeros(shape)
    from_st_last = np.zeros(shape)

    for i in range(n):
        for j in range(n):
            to_st_first[0, i, j, i, j] = 1.0
            to_st_last[-1, i, j, i, j] = 1.0

            from_st_first[i, j, 0, i, j] = 1.0
            from_st_last[i, j, -1, i, j] = 1.0

    to_st_first = to_st_first.reshape((n ** 3, n ** 2))
    to_st_last = to_st_last.reshape((n ** 3, n ** 2))

    from_st_first = from_st_first.reshape((n ** 2, n ** 3))
    from_st_last = from_st_last.reshape((n ** 2, n ** 3))

    assert np.allclose(from_st_first @ to_st_first, np.eye(n ** 2))
    assert np.allclose(from_st_last @ to_st_last, np.eye(n ** 2))

    assert np.allclose(from_st_first @ to_st_last, 0.0)
    assert np.allclose(from_st_last @ to_st_first, 0.0)

    return to_st_first, to_st_last, from_st_first, from_st_last, xp_to_xm, yp_to_ym


def three_stage_von_neumann(x_cfl, y_cfl, x_shifts, y_shifts, solver, batch_size=10_000):
    (Dt, Dx, Dy, volume_integral, first_space_integral,
     last_space_integral, xm_integral, xp_integral, ym_integral, yp_integral) = solver.get_matrices()
    to_st_first, to_st_last, from_st_first, from_st_last, xp_to_xm, yp_to_ym = get_matrices(solver.poly_order)

    M1 = Dt + x_cfl * Dx + y_cfl * Dy + first_space_integral
    M1_inv = np.linalg.inv(M1)

    use_inv = False

    #### mat inverses
    M = M1 + x_cfl * (xm_integral - xp_integral) + y_cfl * (ym_integral - yp_integral)
    Mx = M1 + x_cfl * (xm_integral - xp_integral)
    My = M1 + y_cfl * (ym_integral - yp_integral)
    M_inv = np.linalg.inv(M)
    Mx_inv = np.linalg.inv(Mx)
    My_inv = np.linalg.inv(My)

    first_space_integral = first_space_integral @ to_st_first

    #### first stage
    state_pred_1 = M1_inv @ first_space_integral

    eigs_out = []
    for i in range(0, x_shifts.size, batch_size):

        j = min(i + batch_size, x_shifts.size)

        ### second stage
        R1 = first_space_integral - x_cfl * xp_integral @ state_pred_1
        R2 = (x_cfl * xm_integral @ xp_to_xm @ state_pred_1) * np.exp(-x_shifts[i:j])[:, None, None]
        R = R1 + R2
        if use_inv:
            state_pred_2 = Mx_inv @ R
        else:  
            shape = R.shape
            R = R.swapaxes(0, 1).reshape((shape[1], -1))
            state_pred_2 = scipy.linalg.solve(Mx, R).reshape((shape[1], shape[0], shape[2])).swapaxes(0, 1)

        # ### third stage
        R1 = first_space_integral - y_cfl * yp_integral @ state_pred_1
        R3 = y_cfl * ym_integral @ yp_to_ym @ state_pred_1 * np.exp(-y_shifts[i:j])[:, None, None]
        R = R1 + R3
        if use_inv:
            state_pred_3 = My_inv @ R
        else:
            shape = R.shape
            R = R.swapaxes(0, 1).reshape((shape[1], -1))
            state_pred_3 = scipy.linalg.solve(My, R).reshape((shape[1], shape[0], shape[2])).swapaxes(0, 1)

        # ### final stage
        R1 = first_space_integral - x_cfl * xp_integral @ state_pred_3 - y_cfl * yp_integral @ state_pred_2
        R2 = x_cfl * xm_integral @ xp_to_xm @ state_pred_3 * np.exp(-x_shifts[i:j])[:, None, None]
        R3 = y_cfl * ym_integral @ yp_to_ym @ state_pred_2 * np.exp(-y_shifts[i:j])[:, None, None]
        R = R1 + R2 + R3
        if use_inv:
            mat = M_inv @ R
        else:
            shape = R.shape
            R = R.swapaxes(0, 1).reshape((shape[1], -1))
            mat = scipy.linalg.solve(M, R).reshape((shape[1], shape[0], shape[2])).swapaxes(0, 1)
        
        mat = from_st_last @ mat

        eigs = np.linalg.eigvals(mat)
        eigs_out.extend(eigs)

    return np.array(eigs_out)


def three_stage_von_neumann_fast(x_cfl, y_cfl, x_shifts, y_shifts, solver, batch_size=10_000):
    (Dt, Dx, Dy, volume_integral, first_space_integral,
     last_space_integral, xm_integral, xp_integral, ym_integral, yp_integral) = solver.get_matrices()
    to_st_first, to_st_last, from_st_first, from_st_last, xp_to_xm, yp_to_ym = get_matrices(solver.poly_order)

    M1 = Dt + x_cfl * Dx + y_cfl * Dy + first_space_integral

    use_inv = False

    #### mat inverses
    M = M1 + x_cfl * (xm_integral - xp_integral) + y_cfl * (ym_integral - yp_integral)
    Mx = M1 + x_cfl * (xm_integral - xp_integral)
    My = M1 + y_cfl * (ym_integral - yp_integral)
    
    if use_inv:
        M1_inv = np.linalg.inv(M1)
        M_inv = np.linalg.inv(M)
        Mx_inv = np.linalg.inv(Mx)
        My_inv = np.linalg.inv(My)

    first_space_integral = first_space_integral @ to_st_first

    #### first stage
    if use_inv:
        state_pred_1 = M1_inv @ first_space_integral
    else:
        state_pred_1 = scipy.linalg.solve(M1, first_space_integral)
    
    eigs_out = []
    for i in range(0, x_shifts.size, batch_size):

        j = min(i + batch_size, x_shifts.size)

        exp_x_shifts = np.exp(-x_shifts[i:j])[:, None, None]
        exp_y_shifts = np.exp(-y_shifts[i:j])[:, None, None]

        ### second stage
        R2_0 = first_space_integral - x_cfl * xp_integral @ state_pred_1
        R2_x = (x_cfl * xm_integral @ xp_to_xm @ state_pred_1)
        
        if use_inv:
            state_pred_2_0 = Mx_inv @ R2_0
            state_pred_2_x = Mx_inv @ R2_x
        else:
            state_pred_2_0 = scipy.linalg.solve(Mx, R2_0)
            state_pred_2_x = scipy.linalg.solve(Mx, R2_x)

        # ### third stage
        R3_0 = first_space_integral - y_cfl * yp_integral @ state_pred_1
        R3_y = y_cfl * ym_integral @ yp_to_ym @ state_pred_1

        if use_inv:
            state_pred_3_0 = My_inv @ R3_0
            state_pred_3_y = My_inv @ R3_y
        else:
            state_pred_3_0 = scipy.linalg.solve(My, R3_0)
            state_pred_3_y = scipy.linalg.solve(My, R3_y)

        # ### final stage
        R4_0 = first_space_integral - x_cfl * xp_integral @ state_pred_3_0 - y_cfl * yp_integral @ state_pred_2_0
        R4_x = -y_cfl * yp_integral @ state_pred_2_x + x_cfl * xm_integral @ xp_to_xm @ state_pred_3_0
        R4_y = -x_cfl * xp_integral @ state_pred_3_y + y_cfl * ym_integral @ yp_to_ym @ state_pred_2_0
        R4_xy = y_cfl * ym_integral @ yp_to_ym @ state_pred_2_x + x_cfl * xm_integral @ xp_to_xm @ state_pred_3_y

        if use_inv:
            M_inv = from_st_last @ M_inv
            mat_0 = M_inv @ R4_0
            mat_x = M_inv @ R4_x
            mat_y = M_inv @ R4_y
            mat_xy = M_inv @ R4_xy
        else:
            mat_0 = from_st_last @ scipy.linalg.solve(M, R4_0)
            mat_x = from_st_last @ scipy.linalg.solve(M, R4_x)
            mat_y = from_st_last @ scipy.linalg.solve(M, R4_y)
            mat_xy = from_st_last @ scipy.linalg.solve(M, R4_xy)
        
        mat = mat_0 + mat_x * exp_x_shifts + mat_y * exp_y_shifts + mat_xy * exp_x_shifts * exp_y_shifts
        
        eigs = np.linalg.eigvals(mat)
        eigs_out.extend(eigs)

    return np.array(eigs_out)

orders = [3, 4, 5, 6, 7, 8, 9]
cfls = [
    0.0, 0.2, 0.4,
    0.55, 0.60, 0.65, 0.685, 0.686, 0.687, 0.688, 0.689, 0.69, 0.691, 0.692, 0.693, 0.694, 0.695, 0.696, 0.697, 0.698, 
    0.699, 0.7, 0.701, 0.702, 0.703, 0.704, 0.705, 0.71, 0.72, 0.73
 ]

if run:

    for poly_order in orders:

        shifts = np.linspace(0.0, 2 * np.pi, nk) * 1.0j
        x_shifts, y_shifts = np.meshgrid(shifts, shifts)
        x_shifts = x_shifts.ravel()
        y_shifts = y_shifts.ravel()

        mask = y_shifts <= x_shifts
        x_shifts = x_shifts[mask][rank::ncpus]
        y_shifts = y_shifts[mask][rank::ncpus]

        if rank == 0:
            print(f"Running order {poly_order}")
            print(len(x_shifts))

        solver = BaseADERDG2D(xlim=1.0, ylim=1.0, nx=3, ny=3, poly_order=poly_order)

        top_mags = []
        for cfl in cfls:
            x_cfl = y_cfl = cfl / np.sqrt(2)

            eigs = three_stage_von_neumann_fast(x_cfl, y_cfl, x_shifts, y_shifts, solver=solver)

            top_mags.append(abs(eigs).max())

        top_mags = np.array(top_mags)

        suffix = f'order_{poly_order}_rank_{rank}_of_{ncpus}_nk_{nk}.npy'
        fp = os.path.join(data_dir, f'top_mags_{suffix}')
        np.save(fp, top_mags)

if plot:
    plt.figure()
    for poly_order in orders:

        top_mags_list = []
        for rank in range(ncpus):
            suffix = f'order_{poly_order}_rank_{rank}_of_{ncpus}_nk_{nk}.npy'
            fp = os.path.join(data_dir, f'top_mags_{suffix}')
            top_mags_list.append(np.load(fp))

        top_mags = np.array(top_mags_list).max(axis=0)

        plt.semilogy(cfls, top_mags, '-', label=f'Order {poly_order}')

        print(poly_order)
        print(cfls)
        print(top_mags)
        print()

    # plt.semilogy(cfls, np.ones_like(top_mags) + 1e-6, '--')
    plt.ylabel("Amplification factor")
    plt.xlabel("CFL")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "new-ader-dg-2D-amplification.png"))

    plt.figure()
    for poly_order in orders:

        top_mags_list = []
        for rank in range(ncpus):
            suffix = f'order_{poly_order}_rank_{rank}_of_{ncpus}_nk_{nk}.npy'
            fp = os.path.join(data_dir, f'top_mags_{suffix}')
            top_mags_list.append(np.load(fp))

        top_mags = np.array(top_mags_list).max(axis=0)

        plt.plot(cfls, top_mags - 1.0, '-', label=f'Order {poly_order}')
        plt.yscale('symlog', linthresh=1e-14)

        print(poly_order)
        print(cfls)
        print(top_mags)
        print()

    # plt.semilogy(cfls, np.ones_like(top_mags) + 1e-6, '--')
    plt.ylabel("Amplification factor")
    plt.xlabel("CFL")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "new-ader-dg-2D-amplification-minus-one.png"))

    plt.show()
