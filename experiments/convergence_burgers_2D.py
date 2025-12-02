from ader_dg_transport.ader_dg_2D.burgers_ader_dg_2D import BurgersAderDG2D
import time
import numpy as np
import os
from matplotlib import pyplot as plt


exp_name = 'convergence-burgers-2D'

data_dir = f'data/{exp_name}'
plot_dir = f'plots/{exp_name}'


if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def initial_condition(x, y):
    return 0.25 * (1 - np.cos(x)) * (1 - np.cos(y))


def exact_solution(x, y, t):
    x0 = x - t
    y0 = y - t

    for _ in range(50):
        u = initial_condition(x0, y0)
        x0_ = x - t * u
        y0_ = y - t * u

        diff = max(abs(x0_ - x0).max(), abs(y0_ - y0).max())

        x0 = x0_
        y0 = y0_

        if diff < 1e-15:
            break

    u = initial_condition(x0, y0)

    if diff > 1e-14:
        print('Lagrange diff:', diff)

    return u


xlim = ylim = 2 * np.pi
tend = 0.4
cfl = 0.6
run = True
plot = True

poly_orders = [3, 5, 7]


if run:
    for poly_order in poly_orders:
        if poly_order <= 5:
            nxs = [11, 22, 33, 44]
        else:
            nxs = [6, 8, 12, 16]

        for nx in nxs:

            ny = nx
            dx = xlim / nx
            vel = np.sqrt(2)
            dt = cfl * dx / vel
            n_steps = int(tend / dt) + 1

            out_fp = os.path.join(data_dir, f'u_nx{nx}_ny{ny}_p{poly_order}_t{tend}.npy')

            print(f'Order={poly_order}, nx={nx}')
            print(f'Timestep: {dt}. Nsteps: {n_steps}.')

            solver = BurgersAderDG2D(
                xlim, ylim, nx, ny, poly_order=poly_order, dt=dt
            )

            xs, ys = solver.xs[:, :, 0], solver.ys[:, :, 0]
            solver.u[:] = initial_condition(xs, ys)
            norm0 = solver.norm(solver.u)

            t0 = time.time()

            while solver.time < tend:
                dt = min(solver.dt, tend - solver.time)
                solver.dt = dt
                solver.time_step(tol=1e-14, verbose=False)

            t1 = time.time()

            norm1 = solver.norm(solver.u)

            print('Wall time:', t1 - t0, 's')
            print('Simulation time:', solver.time, 's')
            print('Relative norm change:', (norm1 - norm0) / norm0)

            u_exact = exact_solution(xs, ys, solver.time)
            error = solver.norm(solver.u - u_exact) / solver.norm(u_exact)
            print(f'nx={nx} relative L2 error:', error)
            print()

            np.save(out_fp, solver.u)

if plot:
    for poly_order in poly_orders:
        if poly_order <= 5:
            nxs = [11, 22, 33, 44]
        else:
            nxs = [6, 8, 12, 16]

        errors = []
        norm_change = []
        for nx in nxs:

            ny = nx
            dt = 0.0

            out_fp = os.path.join(data_dir, f'u_nx{nx}_ny{ny}_p{poly_order}_t{tend}.npy')

            solver = BurgersAderDG2D(
                xlim, ylim, nx, ny, poly_order=poly_order, dt=dt)

            xs, ys = solver.xs[:, :, 0], solver.ys[:, :, 0]

            solver.u[:] = np.load(out_fp)

            u_exact = exact_solution(xs, ys, tend)
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
        print(f'p={poly_order} nx={nx}')
        print(f'Convergence:', convergence)
        print('Relative error:', errors)
        print(f'Relative norm change:', norm_change)
        print()
        plt.loglog(xlim / nxs, errors, '-*', label=f'Order {poly_order}')
        plt.loglog(xlim / nxs, 1.5 * errors[0] * (nxs[0] / nxs) ** poly_order, '--', label=f'$\Delta x^{poly_order}$')

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

