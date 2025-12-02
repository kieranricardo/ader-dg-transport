import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
from ader_dg_transport.utils import gll, lagrange1st
from matplotlib import pyplot as plt
from scipy.linalg import lu_factor, lu_solve
from ader_dg_transport.ader_dg_3D.base_ader_dg_3D import BaseADERDG3D
from ader_dg_transport.ader_dg_3D.advection_von_neumann_3D import von_neumann_3D
import scipy
import time

from mpi4py import MPI
import argparse

rank = MPI.COMM_WORLD.Get_rank()

parser = argparse.ArgumentParser()
parser.add_argument('--ncfls', type=int, help='Number of cfl numbers')
parser.add_argument('--nk', type=int, help='Number of wave numbers')
parser.add_argument('--ncpus', type=int, help='Number of procs', default=1)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

ncfls = args.ncfls
nk = args.nk
ncpus = args.ncpus
run = (not args.plot)
plot = args.plot

data_dir = 'data/stability_3D'
plot_dir = f'plots'
# plot_dir = '../../../latex/ADER Transport/plots'

if rank == 0:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

orders = [3, 4, 5, 6]

cfls = [0.0, 0.2, 0.4, 0.55, 0.56] + list(np.linspace(0.57, 0.59, ncfls)) + [0.60,]

if run:

    for poly_order in orders:

        solver = BaseADERDG3D(xlim=1.0, ylim=1.0, zlim=1.0, nx=3, ny=3, nz=3, poly_order=poly_order)

        shifts = np.linspace(0.0, 2 * np.pi, nk) * 1.0j
        x_shifts, y_shifts, z_shifts = np.meshgrid(shifts, shifts, shifts)

        x_shifts = x_shifts.ravel()
        y_shifts = y_shifts.ravel()
        z_shifts = z_shifts.ravel()

        mask = (y_shifts <= x_shifts) & (z_shifts <= y_shifts)

        x_shifts = x_shifts[mask][rank::ncpus]
        y_shifts = y_shifts[mask][rank::ncpus]
        z_shifts = z_shifts[mask][rank::ncpus]

        if rank == 0:
            print(f"Running order {poly_order}")
            print(len(x_shifts))

        top_mags = []
        for cfl in cfls:
            x_cfl = y_cfl = z_cfl = cfl / np.sqrt(3)
            eigs = von_neumann_3D(
                solver, x_cfl, y_cfl, z_cfl, x_shifts, y_shifts, z_shifts, use_power_iteration=False, verbose=False, batch_size=100
            )

            top_mags.append(abs(eigs).max())

        top_mags = np.array(top_mags)

        suffix = f'order_{poly_order}_rank_{rank}_of_{ncpus}_ncfls_{ncfls}_nk_{nk}.npy'
        fp = os.path.join(data_dir, f'top_mags_{suffix}')
        np.save(fp, top_mags)


if plot:
    plt.figure()
    for poly_order in orders:

        top_mags_list = []
        for rank in range(ncpus):
            suffix = f'order_{poly_order}_rank_{rank}_of_{ncpus}_ncfls_{ncfls}_nk_{nk}.npy'
            fp = os.path.join(data_dir, f'top_mags_{suffix}')
            top_mags_list.append(np.load(fp))

        top_mags = np.array(top_mags_list).max(axis=0)

        plt.semilogy(cfls, top_mags, label=f'Order {poly_order}')

    # plt.semilogy(cfls, np.ones_like(top_mags) + 1e-6, '--')
    plt.ylabel("Amplification factor")
    plt.xlabel("CFL")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "new-ader-dg-3D-amplification.png"))

    plt.figure()
    for poly_order in orders:

        top_mags_list = []
        for rank in range(ncpus):
            suffix = f'order_{poly_order}_rank_{rank}_of_{ncpus}_ncfls_{ncfls}_nk_{nk}.npy'
            fp = os.path.join(data_dir, f'top_mags_{suffix}')
            top_mags_list.append(np.load(fp))

        top_mags = np.array(top_mags_list).max(axis=0)

        plt.plot(cfls, top_mags - 1.0, '-', label=f'Order {poly_order}')
        plt.yscale('symlog', linthresh=1e-12)

    # plt.semilogy(cfls, np.ones_like(top_mags) + 1e-6, '--')
    plt.ylabel("Amplification factor")
    plt.xlabel("CFL")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(plot_dir, "new-ader-dg-3D-amplification-minus-one.png"))

    plt.show()