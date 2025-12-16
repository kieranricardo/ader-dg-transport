import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from ader_dg_transport.ader_dg_3D.base_ader_dg_3D import BaseADERDG3D

from mpi4py import MPI
import argparse
import time

EPS = 1e-4
comm = MPI.COMM_WORLD

parser = argparse.ArgumentParser()
parser.add_argument('--o', type=int, help='Polynomial order')
parser.add_argument('--nk', type=int, help='Number of wave numbers')
args = parser.parse_args()

order = args.o
nk = args.nk
ncpus = comm.Get_size()
rank = comm.Get_rank()


def get_matrices(poly_order):
    n = poly_order + 1

    shape = (n,) * 8

    xp_to_xm = np.zeros(shape)
    yp_to_ym = np.zeros(shape)
    zp_to_zm = np.zeros(shape)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                xp_to_xm[i, 0, j, k, i, -1, j, k] = 1.0
                yp_to_ym[i, j, 0, k, i, j, -1, k] = 1.0
                zp_to_zm[i, j, k, 0, i, j, k, -1] = 1.0

    assert np.allclose(xp_to_xm, yp_to_ym.swapaxes(1, 2).swapaxes(5, 6))
    assert np.allclose(xp_to_xm, zp_to_zm.swapaxes(1, 3).swapaxes(5, 7))

    xp_to_xm = xp_to_xm.reshape((n ** 4, n ** 4))
    yp_to_ym = yp_to_ym.reshape((n ** 4, n ** 4))
    zp_to_zm = zp_to_zm.reshape((n ** 4, n ** 4))

    xm_to_xp = xp_to_xm.transpose().copy()
    ym_to_yp = yp_to_ym.transpose().copy()
    zm_to_zp = zp_to_zm.transpose().copy()

    shape = (n,) * 7
    to_st_first = np.zeros(shape)
    to_st_last = np.zeros(shape)

    from_st_first = np.zeros(shape)
    from_st_last = np.zeros(shape)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                to_st_first[0, i, j, k, i, j, k] = 1.0
                to_st_last[-1, i, j, k, i, j, k] = 1.0

                from_st_first[i, j, k, 0, i, j, k] = 1.0
                from_st_last[i, j, k, -1, i, j, k] = 1.0

    to_st_first = to_st_first.reshape((n ** 4, n ** 3))
    to_st_last = to_st_last.reshape((n ** 4, n ** 3))

    from_st_first = from_st_first.reshape((n ** 3, n ** 4))
    from_st_last = from_st_last.reshape((n ** 3, n ** 4))

    assert np.allclose(from_st_first @ to_st_first, np.eye(n ** 3))
    assert np.allclose(from_st_last @ to_st_last, np.eye(n ** 3))

    assert np.allclose(from_st_first @ to_st_last, 0.0)
    assert np.allclose(from_st_last @ to_st_first, 0.0)

    return to_st_first, to_st_last, from_st_first, from_st_last, xp_to_xm, yp_to_ym, zp_to_zm, xm_to_xp, ym_to_yp, zm_to_zp


def von_neumann_analysis(solver, cfl, niter, x_shifts, y_shifts, z_shifts, batch_size=10_000):

    t0 = time.time()
    nvars = 4
    dim = 4

    x_cfl = cfl
    y_cfl = cfl
    z_cfl = cfl

    (Dt, Dx, Dy, Dz, first_space_integral,
     last_space_integral, xm_integral, xp_integral,
     ym_integral, yp_integral, zm_integral, zp_integral) = solver.get_matrices()
    to_st_first, to_st_last, from_st_first, from_st_last, xp_to_xm, yp_to_ym, zp_to_zm, xm_to_xp, ym_to_yp, zm_to_zp = get_matrices(solver.poly_order)

    n = solver.poly_order + 1
    sz = n ** dim
    u_slice = slice(0, sz)
    v_slice = slice(sz, 2 * sz)
    w_slice = slice(2 * sz, 3 * sz)
    h_slice = slice(3 * sz, 4 * sz)

    M1 = np.zeros((nvars * sz, nvars * sz))
    M1[u_slice, u_slice] += Dt + first_space_integral
    M1[v_slice, v_slice] += Dt + first_space_integral
    M1[w_slice, w_slice] += Dt + first_space_integral
    M1[h_slice, h_slice] += Dt + first_space_integral
    # dudt + dhdx = 0
    M1[u_slice, h_slice] += x_cfl * Dx
    # dvdt + dhdy = 0
    M1[v_slice, h_slice] += y_cfl * Dy
    # dwdt + dhdz = 0
    M1[w_slice, h_slice] += z_cfl * Dz

    # dhdt + dudx + dvdy + dwdy = 0
    M1[h_slice, u_slice] += x_cfl * Dx
    M1[h_slice, v_slice] += y_cfl * Dy
    M1[h_slice, w_slice] += z_cfl * Dz

    M2 = np.copy(M1)
    # dudt + dhdx = 0
    M2[u_slice, h_slice] += 0.5 * x_cfl * (xm_integral - xp_integral)
    # dvdt + dhdy = 0
    M2[v_slice, h_slice] += 0.5 * y_cfl * (ym_integral - yp_integral)
    # dwdt + dhdz = 0
    M2[w_slice, h_slice] += 0.5 * z_cfl * (zm_integral - zp_integral)
    # dhdt + dudx + dvdy + dwdz = 0
    M2[h_slice, u_slice] += 0.5 * x_cfl * (xm_integral - xp_integral)
    M2[h_slice, v_slice] += 0.5 * y_cfl * (ym_integral - yp_integral)
    M2[h_slice, w_slice] += 0.5 * z_cfl * (zm_integral - zp_integral)

    # dissipation terms
    # 0.5 * c * u * dy * dt
    M2[u_slice, u_slice] += 0.5 * x_cfl * (xm_integral + xp_integral)
    M2[v_slice, v_slice] += 0.5 * y_cfl * (ym_integral + yp_integral)
    M2[w_slice, w_slice] += 0.5 * z_cfl * (zm_integral + zp_integral)
    M2[h_slice, h_slice] += 0.5 * x_cfl * (xm_integral + xp_integral)
    M2[h_slice, h_slice] += 0.5 * y_cfl * (ym_integral + yp_integral)
    M2[h_slice, h_slice] += 0.5 * z_cfl * (zm_integral + zp_integral)

    # M1_inv = np.linalg.inv(M1)
    # M2_inv = np.linalg.inv(M2)

    to_st_first_all_vars = np.zeros((nvars * sz, nvars * n ** (dim - 1)))
    for i in range(nvars):
        to_st_first_all_vars[i * sz:(i + 1) * sz, i * n ** (dim - 1):(i + 1) * n ** (dim - 1)] = to_st_first

    from_st_last_all_vars = np.zeros((nvars * n ** (dim - 1), nvars * sz))
    for i in range(nvars):
        from_st_last_all_vars[i * n ** (dim - 1):(i + 1) * n ** (dim - 1), i * sz:(i + 1) * sz] = from_st_last

    xp_to_xm_all_vars = np.zeros((nvars * sz, nvars * sz))
    for i in range(nvars):
        xp_to_xm_all_vars[i * sz:(i + 1) * sz, i * sz:(i + 1) * sz] = xp_to_xm

    xm_to_xp_all_vars = np.zeros((nvars * sz, nvars * sz))
    for i in range(nvars):
        xm_to_xp_all_vars[i * sz:(i + 1) * sz, i * sz:(i + 1) * sz] = xm_to_xp

    yp_to_ym_all_vars = np.zeros((nvars * sz, nvars * sz))
    for i in range(nvars):
        yp_to_ym_all_vars[i * sz:(i + 1) * sz, i * sz:(i + 1) * sz] = yp_to_ym

    ym_to_yp_all_vars = np.zeros((nvars * sz, nvars * sz))
    for i in range(nvars):
        ym_to_yp_all_vars[i * sz:(i + 1) * sz, i * sz:(i + 1) * sz] = ym_to_yp

    zp_to_zm_all_vars = np.zeros((nvars * sz, nvars * sz))
    for i in range(nvars):
        zp_to_zm_all_vars[i * sz:(i + 1) * sz, i * sz:(i + 1) * sz] = zp_to_zm

    zm_to_zp_all_vars = np.zeros((nvars * sz, nvars * sz))
    for i in range(nvars):
        zm_to_zp_all_vars[i * sz:(i + 1) * sz, i * sz:(i + 1) * sz] = zm_to_zp

    t_bdry = np.zeros((nvars * sz, nvars * sz))
    t_bdry[u_slice, u_slice] += first_space_integral
    t_bdry[v_slice, v_slice] += first_space_integral
    t_bdry[w_slice, w_slice] += first_space_integral
    t_bdry[h_slice, h_slice] += first_space_integral

    xm_bdry = np.zeros_like(t_bdry)
    xp_bdry = np.zeros_like(t_bdry)
    ym_bdry = np.zeros_like(t_bdry)
    yp_bdry = np.zeros_like(t_bdry)
    zm_bdry = np.zeros_like(t_bdry)
    zp_bdry = np.zeros_like(t_bdry)

    xm_bdry[u_slice, h_slice] += 0.5 * x_cfl * xm_integral
    xm_bdry[h_slice, u_slice] += 0.5 * x_cfl * xm_integral

    xm_bdry[u_slice, u_slice] += 0.5 * x_cfl * xm_integral
    xm_bdry[h_slice, h_slice] += 0.5 * x_cfl * xm_integral

    #############
    xp_bdry[u_slice, h_slice] += 0.5 * x_cfl * (- xp_integral)
    xp_bdry[h_slice, u_slice] += 0.5 * x_cfl * (- xp_integral)

    xp_bdry[u_slice, u_slice] += 0.5 * x_cfl * xp_integral
    xp_bdry[h_slice, h_slice] += 0.5 * x_cfl * xp_integral

    #############
    ym_bdry[v_slice, h_slice] += 0.5 * y_cfl * ym_integral
    ym_bdry[h_slice, v_slice] += 0.5 * y_cfl * ym_integral

    ym_bdry[v_slice, v_slice] += 0.5 * y_cfl * ym_integral
    ym_bdry[h_slice, h_slice] += 0.5 * y_cfl * ym_integral

    #############
    yp_bdry[v_slice, h_slice] += 0.5 * y_cfl * (- yp_integral)
    yp_bdry[h_slice, v_slice] += 0.5 * y_cfl * (- yp_integral)

    yp_bdry[v_slice, v_slice] += 0.5 * y_cfl * yp_integral
    yp_bdry[h_slice, h_slice] += 0.5 * y_cfl * yp_integral

    #############
    zm_bdry[w_slice, h_slice] += 0.5 * z_cfl * zm_integral
    zm_bdry[h_slice, w_slice] += 0.5 * z_cfl * zm_integral

    zm_bdry[w_slice, w_slice] += 0.5 * z_cfl * zm_integral
    zm_bdry[h_slice, h_slice] += 0.5 * z_cfl * zm_integral

    #############
    zp_bdry[w_slice, h_slice] += 0.5 * z_cfl * (- zp_integral)
    zp_bdry[h_slice, w_slice] += 0.5 * z_cfl * (- zp_integral)

    zp_bdry[w_slice, w_slice] += 0.5 * z_cfl * zp_integral
    zp_bdry[h_slice, h_slice] += 0.5 * z_cfl * zp_integral

    ##############
    xm_bdry = xm_bdry @ xp_to_xm_all_vars
    xp_bdry = xp_bdry @ xm_to_xp_all_vars

    ym_bdry = ym_bdry @ yp_to_ym_all_vars
    yp_bdry = yp_bdry @ ym_to_yp_all_vars

    zm_bdry = zm_bdry @ zp_to_zm_all_vars
    zp_bdry = zp_bdry @ zm_to_zp_all_vars

    t1 = time.time()
    # print(f"Setup time: {t1 - t0}")

    arr = np.zeros((nvars * sz, nvars * n ** (dim - 1)))
    state_pred = dict()
    state_pred[(0, 0, 0)] = t_bdry @ to_st_first_all_vars

    for key in state_pred.keys():
        state_pred[key] = np.linalg.solve(M1, state_pred[key])

    for _ in range(niter):

        state_pred_ = dict()
        state_pred_[(0, 0, 0)] = t_bdry @ to_st_first_all_vars

        for key, val in state_pred.items():
            new_key = (key[0] + 1, key[1], key[2])
            state_pred_[new_key] = xm_bdry @ val + state_pred_.get(new_key, arr)
            new_key = (key[0] - 1, key[1], key[2])
            state_pred_[new_key] = xp_bdry @ val + state_pred_.get(new_key, arr)

            new_key = (key[0], key[1] + 1, key[2])
            state_pred_[new_key] = ym_bdry @ val + state_pred_.get(new_key, arr)
            new_key = (key[0], key[1] - 1, key[2])
            state_pred_[new_key] = yp_bdry @ val + state_pred_.get(new_key, arr)

            new_key = (key[0], key[1], key[2] + 1)
            state_pred_[new_key] = zm_bdry @ val + state_pred_.get(new_key, arr)
            new_key = (key[0], key[1], key[2] - 1)
            state_pred_[new_key] = zp_bdry @ val + state_pred_.get(new_key, arr)

        for key, val in state_pred_.items():
            state_pred_[key] = np.linalg.solve(M2, val)

        del state_pred
        state_pred = state_pred_

    t2 = time.time()
    # print(f"Mat time: {t2 - t1}")

    eigs_all = []
    for i in range(0, x_shifts.size, batch_size):

        j = min(i + batch_size, x_shifts.size)

        mat = None
        for key, val in state_pred.items():
            exp_part = np.exp(key[0] * x_shifts) * np.exp(key[1] * y_shifts) * np.exp(key[2] * z_shifts)
            if mat is None:
                mat = (from_st_last_all_vars @ val)[None] * exp_part[:, None, None]
            else:
                mat += (from_st_last_all_vars @ val)[None] * exp_part[:, None, None]

        eigs = np.linalg.eigvals(mat)
        eigs_all.extend(eigs)

    t3 = time.time()
    # print(f"Eig time: {t3 - t2}")
    # print()

    return abs(np.array(eigs_all)).max()


def para_von_neumann_analysis(solver, cfl, niter, nk, batch_size=10_000):
    shifts = np.linspace(0.0, 2 * np.pi, nk) * 1.0j
    x_shifts, y_shifts, z_shifts = np.meshgrid(shifts, shifts, shifts)
    x_shifts = x_shifts.ravel()
    y_shifts = y_shifts.ravel()
    z_shifts = z_shifts.ravel()

    mask = y_shifts <= x_shifts
    mask &= z_shifts <= y_shifts
    x_shifts = x_shifts[mask][rank::ncpus]
    y_shifts = y_shifts[mask][rank::ncpus]
    z_shifts = z_shifts[mask][rank::ncpus]

    amp = von_neumann_analysis(solver, cfl, niter, x_shifts, y_shifts, z_shifts, batch_size=batch_size)
    max_amp = comm.reduce(amp, op=MPI.MAX, root=0)

    max_amp = comm.bcast(max_amp, root=0)

    return max_amp


def max_cfl(solver, niter, nk, batch_size=10_000):

    hi = 2.0
    lo = 1e-6

    for _ in range(10):

        mid = 0.5 * (lo + hi)
        max_amp = para_von_neumann_analysis(solver, mid, niter, nk, batch_size=batch_size)

        if max_amp > (1 + EPS):
            hi = mid
        else:
            lo = mid

    return lo

if rank == 0:
    print(f'Stability threshold = 1 + {EPS}')

# solver = BaseADERDG3D(xlim=1.0, ylim=1.0, zlim=1.0, nx=3, ny=3, nz=3, poly_order=order)
# for niter in range(3, 4):
#
#     cfl = max_cfl(solver, niter, nk=nk)
#     if rank == 0:
#         print(f'Order {solver.poly_order} with {niter} iterations max CFL: {cfl:.5f}. Communication eff: {cfl / niter:.5f}. Compute eff: {cfl / (niter + 1):.5f}')
#
#     comm.barrier()

niter = 3
for poly_order in range(3, order + 1):
    solver = BaseADERDG3D(xlim=1.0, ylim=1.0, zlim=1.0, nx=3, ny=3, nz=3, poly_order=poly_order)
    cfl = max_cfl(solver, niter, nk=nk)
    if rank == 0:
        print(f'Order {solver.poly_order} with {niter} iterations max CFL: {cfl:.5f}. Communication eff: {cfl / niter:.5f}. Compute eff: {cfl / (niter + 1):.5f}')

    comm.barrier()
