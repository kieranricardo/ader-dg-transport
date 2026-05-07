from ader_dg_transport.ader_dg_2D.advection_iter_ader_dg_2D import AdvectionIterAderDG2D
from ader_dg_transport.ader_dg_2D.advection_ader_dg_2D import AdvectionAderDG2D
import time
import numpy as np
import os
from matplotlib import pyplot as plt

exp_name = 'convergence-advection-2D'
data_dir = f'data/{exp_name}'
plot_dir = f'plots/{exp_name}'


if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def initial_condition(x, y):
    return np.sin(16 * np.pi * x) * np.sin(16 * np.pi * y)


xlim = ylim = 2
tend = 2.0
cx = cy = 1.0
cfl = 0.6
run = False
plot = True

if run:

    for poly_order in [3, 5, 7]:

        if poly_order <= 5:
            nxs = [20, 40, 80, 160]
        else:
            nxs = [20, 40, 80]

        for nx in nxs:

            nx = ny = nx
            dx = xlim / nx
            dt = cfl * dx / np.sqrt(cx**2 + cy**2)
            n_steps = int(tend / dt) + 1
            dt = tend / n_steps

            out_fp = os.path.join(data_dir, f'u_nx{nx}_ny{ny}_p{poly_order}_t{tend}.npy')

            solver = AdvectionAderDG2D(
                xlim, ylim, nx, ny, poly_order=poly_order, dt=dt, cx=cx, cy=cy)

            xs, ys = solver.xs[:, :, 0], solver.ys[:, :, 0]
            solver.u[:] = initial_condition(xs, ys)

            t0 = time.time()
            N = 5
            for _ in range(n_steps):
                solver.time_step()
            t1 = time.time()

            print(f'Order={poly_order}, nx={nx}')
            print('Actual CFL:', dt * np.sqrt(cx**2 + cy**2) / dx)
            print('Wall time:', t1 - t0, 's')
            print('Simulation time:', solver.time, 's')
            # print('Total time:', (tend / dt) * (t1 - t0) / 5)
            print(out_fp)
            print()

            np.save(out_fp, solver.u)

if plot:
    poly_orders = [3, 5, 7]
    for poly_order in poly_orders:

        if poly_order <= 5:
            nxs = [20, 40, 80, 160]
        else:
            nxs = [20, 40, 80]

        errors = []
        norm_change = []
        for nx in nxs:

            ny = nx
            dt = 0.0

            out_fp = os.path.join(data_dir, f'u_nx{nx}_ny{ny}_p{poly_order}_t{tend}.npy')

            solver = AdvectionAderDG2D(
                xlim, ylim, nx, ny, poly_order=poly_order, dt=dt, cx=0, cy=0)

            xs, ys = solver.xs[:, :, 0], solver.ys[:, :, 0]

            solver.u[:] = np.load(out_fp)

            xs, ys = solver.xs[:, :, 0], solver.ys[:, :, 0]
            x_ct = (xs - cx * tend) % solver.xlim
            y_ct = (ys - cy * tend) % solver.ylim

            u_exact = initial_condition(x_ct, y_ct)

            error = solver.norm(solver.u - u_exact) / solver.norm(u_exact)
            errors.append(error)

            dnorm = (solver.norm(solver.u) - solver.norm(u_exact)) / solver.norm(u_exact)
            norm_change.append(dnorm)
            # print(f'p={poly_order} nx={nx}')
            # print(f'Relative L2 error:', error)
            # print('norm growth:', (solver.norm(solver.u) - solver.norm(u_exact)) / solver.norm(u_exact))
            # print()

        errors = np.array(errors)
        nxs = np.array(nxs)
        convergence = np.log(errors[:-1] / errors[1:]) / np.log(nxs[1:] / nxs[:-1])
        print(f'p={poly_order}')
        print(f'Convergence:', convergence)
        print('Relative error:', errors)
        print(f'Relative norm change:', norm_change)
        print()
        plt.loglog(xlim / nxs, errors, '-*', label=f'Order {poly_order}')
        plt.loglog(xlim / nxs, 1.5 * errors[0] * (nxs[0] / nxs)**poly_order, '--', label=f'$\Delta x^{poly_order}$')

    plt.ylabel("Relative $L^2$ error")
    plt.xlabel("Element size $\Delta x$")
    plt.tight_layout()
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(plot_dir, f'{exp_name}.png'))

    plt.figure()
    ax = plt.gca()
    im = solver.plot_solution(data=solver.u, ax=ax)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(im, label='q')

    plt.savefig(os.path.join(plot_dir, f'{exp_name}_solution.png'))

    plt.show()

# errors = []
# for nx in nxs:
#
#     nx = ny = nx
#     dx = xlim / nx
#     dt = 0.0
#
#     out_fp = os.path.join(data_dir, f'u_nx{nx}_ny{ny}_p{poly_order}_t{tend}.npy')
#
#     solver = AdvectionAderDG2D(
#         xlim, ylim, nx, ny, poly_order=poly_order, dt=dt, cx=cx, cy=cy)
#
#     solver.u[:] = np.load(out_fp)
#
#     xs, ys = solver.xs[:, :, 0], solver.ys[:, :, 0]
#     x_ct = (xs - cx * tend) % solver.xlim
#     y_ct = (ys - cy * tend) % solver.ylim
#
#     u_exact = initial_condition(x_ct, y_ct)
#
#     error = solver.norm(u_exact - solver.u) / solver.norm(u_exact)
#     errors.append(error)
#     print(f'nx error:', error)
#     print('norm growth:', (solver.norm(solver.u) - solver.norm(u_exact)) / solver.norm(u_exact))
#     print()

