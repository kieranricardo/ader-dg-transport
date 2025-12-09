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


import argparse

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

    shape = (n,) * 6

    xp_to_xm = np.zeros(shape)
    yp_to_ym = np.zeros(shape)
    xm_to_xp = np.zeros(shape)
    ym_to_yp = np.zeros(shape)

    for i in range(n):
        for j in range(n):
            xp_to_xm[i, 0, j, i, -1, j] = 1.0
            yp_to_ym[i, j, 0, i, j, -1] = 1.0

            xm_to_xp[i, -1, j, i, 0, j] = 1.0
            ym_to_yp[i, j, -1, i, j, 0] = 1.0

    xp_to_xm = xp_to_xm.reshape((n ** 3, n ** 3))
    yp_to_ym = yp_to_ym.reshape((n ** 3, n ** 3))

    xm_to_xp = xm_to_xp.reshape((n ** 3, n ** 3))
    ym_to_yp = ym_to_yp.reshape((n ** 3, n ** 3))

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

    return to_st_first, to_st_last, from_st_first, from_st_last, xp_to_xm, yp_to_ym, xm_to_xp, ym_to_yp


def von_neumann_analysis(solver, cfl, x_shifts, y_shifts, batch_size=10_000):
    x_cfl = cfl
    y_cfl = cfl

    (Dt, Dx, Dy, volume_integral, first_space_integral,
     last_space_integral, xm_integral, xp_integral, ym_integral, yp_integral) = solver.get_matrices()
    to_st_first, to_st_last, from_st_first, from_st_last, xp_to_xm, yp_to_ym, xm_to_xp, ym_to_yp = get_matrices(solver.poly_order)

    n = solver.poly_order + 1
    sz = n ** 3
    u_slice = slice(0, sz)
    v_slice = slice(sz, 2 * sz)
    h_slice = slice(2 * sz, 3 * sz)

    M1 = np.zeros((3 * n ** 3, 3 * n ** 3))
    M1[u_slice, u_slice] += Dt + first_space_integral
    M1[v_slice, v_slice] += Dt + first_space_integral
    M1[h_slice, h_slice] += Dt + first_space_integral
    # dudt + dhdx = 0
    M1[u_slice, h_slice] += x_cfl * Dx
    # dvdt + dhdy = 0
    M1[v_slice, h_slice] += y_cfl * Dy
    # dhdt + dudx + dvdy = 0
    M1[h_slice, u_slice] += x_cfl * Dx
    M1[h_slice, v_slice] += y_cfl * Dy

    M2 = np.copy(M1)
    # dudt + dhdx = 0
    M2[u_slice, h_slice] += 0.5 * x_cfl * (xm_integral - xp_integral)
    # dvdt + dhdy = 0
    M2[v_slice, h_slice] += 0.5 * y_cfl * (ym_integral - yp_integral)
    # dhdt + dudx + dvdy = 0
    M2[h_slice, u_slice] += 0.5 * x_cfl * (xm_integral - xp_integral)
    M2[h_slice, v_slice] += 0.5 * y_cfl * (ym_integral - yp_integral)

    # dissipation terms
    # 0.5 * c * u * dy * dt
    M2[u_slice, u_slice] += 0.5 * x_cfl * (xm_integral + xp_integral)
    M2[v_slice, v_slice] += 0.5 * y_cfl * (ym_integral + yp_integral)
    M2[h_slice, h_slice] += 0.5 * x_cfl * (xm_integral + xp_integral)
    M2[h_slice, h_slice] += 0.5 * y_cfl * (ym_integral + yp_integral)

    # exclude numerical fluxes
    Mxy = np.copy(M1)
    Mxy[u_slice, h_slice] += x_cfl * (xm_integral - xp_integral)
    Mxy[v_slice, h_slice] += y_cfl * (ym_integral - yp_integral)
    Mxy[h_slice, u_slice] += x_cfl * (xm_integral - xp_integral)
    Mxy[h_slice, v_slice] += y_cfl * (ym_integral - yp_integral)

    # exclude x-numerical fluxes
    Mx = np.copy(M1)
    Mx[u_slice, h_slice] += x_cfl * (xm_integral - xp_integral)
    Mx[h_slice, u_slice] += x_cfl * (xm_integral - xp_integral)
    
    # exclude y-numerical fluxes
    My = np.copy(M1)
    My[v_slice, h_slice] += y_cfl * (ym_integral - yp_integral)
    My[h_slice, v_slice] += y_cfl * (ym_integral - yp_integral)
    
    # x numerical fluxes only
    Mx_flx = np.zeros((3 * n ** 3, 3 * n ** 3))
    Mx_flx[u_slice, h_slice] += -0.5 * x_cfl * (xm_integral - xp_integral)
    Mx_flx[h_slice, u_slice] += -0.5 * x_cfl * (xm_integral - xp_integral)
    # dissipation terms
    Mx_flx[u_slice, u_slice] += 0.5 * x_cfl * (xm_integral + xp_integral)
    Mx_flx[h_slice, h_slice] += 0.5 * x_cfl * (xm_integral + xp_integral)

    # y numerical fluxes only
    My_flx = np.zeros((3 * n ** 3, 3 * n ** 3))
    My_flx[v_slice, h_slice] += -0.5 * y_cfl * (ym_integral - yp_integral)
    My_flx[h_slice, v_slice] += -0.5 * y_cfl * (ym_integral - yp_integral)
    # dissipation terms
    My_flx[v_slice, v_slice] += 0.5 * y_cfl * (ym_integral + yp_integral)
    My_flx[h_slice, h_slice] += 0.5 * y_cfl * (ym_integral + yp_integral)

    Mxy_flx = Mx_flx + My_flx

    assert np.allclose(Mxy + Mxy_flx, M2)

    # M1_inv = np.linalg.inv(M1)
    # M2_inv = np.linalg.inv(M2)

    to_st_first_all_vars = np.zeros((3 * sz, 3 * n ** 2))
    for i in range(3):
        to_st_first_all_vars[i * sz:(i + 1) * sz, i * n ** 2:(i + 1) * n ** 2] = to_st_first

    from_st_last_all_vars = np.zeros((3 * n ** 2, 3 * sz))
    for i in range(3):
        from_st_last_all_vars[i * n ** 2:(i + 1) * n ** 2, i * sz:(i + 1) * sz] = from_st_last

    xp_to_xm_all_vars = np.zeros((3 * sz, 3 * sz))
    for i in range(3):
        xp_to_xm_all_vars[i * sz:(i + 1) * sz, i * sz:(i + 1) * sz] = xp_to_xm

    yp_to_ym_all_vars = np.zeros((3 * sz, 3 * sz))
    for i in range(3):
        yp_to_ym_all_vars[i * sz:(i + 1) * sz, i * sz:(i + 1) * sz] = yp_to_ym

    xm_to_xp_all_vars = np.zeros((3 * sz, 3 * sz))
    for i in range(3):
        xm_to_xp_all_vars[i * sz:(i + 1) * sz, i * sz:(i + 1) * sz] = xm_to_xp

    ym_to_yp_all_vars = np.zeros((3 * sz, 3 * sz))
    for i in range(3):
        ym_to_yp_all_vars[i * sz:(i + 1) * sz, i * sz:(i + 1) * sz] = ym_to_yp

    t_bdry = np.zeros((3 * n ** 3, 3 * n ** 3))
    t_bdry[u_slice, u_slice] += first_space_integral
    t_bdry[v_slice, v_slice] += first_space_integral
    t_bdry[h_slice, h_slice] += first_space_integral

    xm_bdry = np.zeros((3 * n ** 3, 3 * n ** 3))
    xp_bdry = np.zeros((3 * n ** 3, 3 * n ** 3))
    ym_bdry = np.zeros((3 * n ** 3, 3 * n ** 3))
    yp_bdry = np.zeros((3 * n ** 3, 3 * n ** 3))

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

    arr = np.zeros((n ** 3 * 3, n ** 2 * 3))

    state_pred_0 = dict()
    state_pred_0[(0, 0)] = t_bdry @ to_st_first_all_vars

    for key, val in state_pred_0.items():
        state_pred_0[key] = np.linalg.solve(M1, val)

    # predictor with x fluxes
    state_pred_1 = dict()
    state_pred_1[(0, 0)] = t_bdry @ to_st_first_all_vars

    for key, val in state_pred_0.items():
        state_pred_1[key] = -Mx_flx @ val + state_pred_1.get(key, arr)
        state_pred_1[(key[0] + 1, key[1])] = xm_bdry @ xp_to_xm_all_vars @ val + state_pred_1.get((key[0] + 1, key[1]), arr)
        state_pred_1[(key[0] - 1, key[1])] = xp_bdry @ xm_to_xp_all_vars @ val + state_pred_1.get((key[0] - 1, key[1]), arr)

    for key, val in state_pred_1.items():
        state_pred_1[key] = np.linalg.solve(Mx, val)
        
    # predictor with y fluxes
    state_pred_2 = dict()
    state_pred_2[(0, 0)] = t_bdry @ to_st_first_all_vars

    for key, val in state_pred_0.items():
        state_pred_2[key] = -My_flx @ val + state_pred_2.get(key, arr)
        state_pred_2[(key[0], key[1] + 1)] = ym_bdry @ yp_to_ym_all_vars @ val + state_pred_2.get((key[0], key[1] + 1), arr)
        state_pred_2[(key[0], key[1] - 1)] = yp_bdry @ ym_to_yp_all_vars @ val + state_pred_2.get((key[0], key[1] - 1), arr)

    for key, val in state_pred_2.items():
        state_pred_2[key] = np.linalg.solve(My, val)
        
    # corrector 
    state_pred = dict()
    state_pred[(0, 0)] = t_bdry @ to_st_first_all_vars

    for key, val in state_pred_1.items():
        state_pred[key] = -My_flx @ val + state_pred.get(key, arr)
        state_pred[(key[0], key[1] + 1)] = ym_bdry @ yp_to_ym_all_vars @ val + state_pred.get((key[0], key[1] + 1), arr)
        state_pred[(key[0], key[1] - 1)] = yp_bdry @ ym_to_yp_all_vars @ val + state_pred.get((key[0], key[1] - 1), arr)

    for key, val in state_pred_2.items():
        state_pred[key] = -Mx_flx @ val + state_pred.get(key, arr)
        state_pred[(key[0] + 1, key[1])] = xm_bdry @ xp_to_xm_all_vars @ val + state_pred.get((key[0] + 1, key[1]), arr)
        state_pred[(key[0] - 1, key[1])] = xp_bdry @ xm_to_xp_all_vars @ val + state_pred.get((key[0] - 1, key[1]), arr)

    for key, val in state_pred.items():
        state_pred[key] = np.linalg.solve(Mxy, val)

    eigs_all = []
    for i in range(0, x_shifts.size, batch_size):

        j = min(i + batch_size, x_shifts.size)
        
        mat = None
        for key, val in state_pred.items():
            if mat is None:
                mat = (from_st_last_all_vars @ val)[None] * (np.exp(key[0] * x_shifts) * np.exp(key[1] * y_shifts))[:, None, None]
            else:
                mat += (from_st_last_all_vars @ val)[None] * (np.exp(key[0] * x_shifts) * np.exp(key[1] * y_shifts))[:, None, None]

        eigs = np.linalg.eigvals(mat)
        eigs_all.extend(eigs)

    return abs(np.array(eigs_all)).max()


def para_von_neumann_analysis(solver, cfl, nk, batch_size=10_000):
    shifts = np.linspace(0.0, 2 * np.pi, nk) * 1.0j
    x_shifts, y_shifts = np.meshgrid(shifts, shifts)
    x_shifts = x_shifts.ravel()
    y_shifts = y_shifts.ravel()

    mask = y_shifts <= x_shifts
    x_shifts = x_shifts[mask][rank::ncpus]
    y_shifts = y_shifts[mask][rank::ncpus]

    amp = von_neumann_analysis(solver, cfl, x_shifts, y_shifts, batch_size=batch_size)
    max_amp = comm.reduce(amp, op=MPI.MAX, root=0)

    max_amp = comm.bcast(max_amp, root=0)

    return max_amp


def max_cfl(solver, nk, batch_size=10_000):

    hi = 2.0
    lo = 1e-6

    for _ in range(10):

        mid = 0.5 * (lo + hi)
        max_amp = para_von_neumann_analysis(solver, mid, nk, batch_size=batch_size)

        if max_amp > (1 + EPS):
            hi = mid
        else:
            lo = mid

    return lo

if rank == 0:
    print(f'Stability threshold = 1 + {EPS}')

solver = BaseADERDG2D(xlim=1.0, ylim=1.0, nx=3, ny=3, poly_order=order)
cfl = max_cfl(solver, nk=nk)
if rank == 0:
    print(f'Order {solver.poly_order} max CFL: {cfl:.5f}. Communication eff: {cfl / 2:.5f}. Compute eff: {cfl / 4:.5f}')

comm.barrier()
