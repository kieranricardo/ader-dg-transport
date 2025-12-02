from ader_dg_transport.ader_dg_3D.advection_iter_ader_dg_3D import AdvectionIterAderDG3D
from ader_dg_transport.ader_dg_3D.advection_ader_dg_3D import AdvectionAderDG3D
import time
import numpy as np
from matplotlib import pyplot as plt
import os


exp_name = 'convergence-advection-3D'

data_dir = f'data/{exp_name}'
plot_dir = f'plots/{exp_name}'
restart_dir = 'data/convergence-advection-3D/restarts'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

if not os.path.exists(restart_dir):
    os.makedirs(restart_dir)


def initial_condition(x, y, z):
    return np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) * np.sin(2 * np.pi * z)


xlim = ylim = zlim = 2
tend = 2.0
cx = cy = cz = 1.0
cfl = 0.55
run = True
plot = True


if run:
    for poly_order in [3, 5, 7]:
        if poly_order <= 5:
            nxs = [5, 10, 20, 40]
        else:
            nxs = [5, 10, 20]
    
        for nx in nxs:

            ny = nz = nx
            dx = xlim / nx
            dt = cfl * dx / np.sqrt(cx**2 + cy**2 + cz**2)
            n_steps = int(tend / dt) + 1
            dt = tend / n_steps

            out_fp = os.path.join(data_dir, f'u_nx{nx}_ny{ny}_nz{nz}_p{poly_order}_t{tend}.npy')

            print(f'Order={poly_order}, nx={nx}')
            print('Actual CFL:', dt * np.sqrt(cx ** 2 + cy ** 2 + cz ** 2) / dx)
            print('dt:', dt)
            print('nteps:', n_steps)
            print(out_fp)

            solver = AdvectionAderDG3D(
                xlim, ylim, zlim, nx, ny, nz, poly_order=poly_order, dt=dt, cx=cx, cy=cy, cz=cz)

            xs, ys, zs = solver.xs[:, :, :, 0], solver.ys[:, :, :, 0], solver.zs[:, :, :, 0]
            solver.u[:] = initial_condition(xs, ys, zs)

            print('Running simulation')
            t0 = time.time()
            for i in range(n_steps):
                if (i % 10) == 0:
                    print('Step:', i)
                    tmp_fp = os.path.join(restart_dir, f'u_nx{nx}_ny{ny}_nz{nz}_p{poly_order}_t{tend}_step_{i}.npy')
                    np.save(tmp_fp, solver.u)
                    
                solver.time_step()      

            t1 = time.time()
            print('Wall time:', (t1 - t0), 's')
            print('Simulation time:', solver.time, 's')
            print()

            np.save(out_fp, solver.u)

if plot:
    for poly_order in [3, 5, 7]:
        
        if poly_order == 3:
             nxs = [5, 10, 20, 40]
        elif poly_order <= 5:
            nxs = [5, 10, 20, 40]
        else:
            nxs = [5, 10, 20]
        
        errors = []
        norm_change = []
        
        for nx in nxs:

            ny = nz = nx
            dt = 0.0

            out_fp = os.path.join(data_dir, f'u_nx{nx}_ny{ny}_nz{nz}_p{poly_order}_t{tend}.npy')

            solver = AdvectionIterAderDG3D(
                xlim, ylim, zlim, nx, ny, nz, poly_order=poly_order, dt=dt, cx=cx, cy=cy, cz=cz)

            solver.u[:] = np.load(out_fp)

            xs, ys, zs = solver.xs[:, :, :, 0], solver.ys[:, :, :, 0], solver.zs[:, :, :, 0]
            x_ct = (xs - cx * tend) % solver.xlim
            y_ct = (ys - cy * tend) % solver.ylim
            z_ct = (zs - cz * tend) % solver.zlim

            u_exact = initial_condition(x_ct, y_ct, z_ct)

            error = solver.norm(u_exact - solver.u) / solver.norm(u_exact)
            errors.append(error)

            dnorm = (solver.norm(solver.u) - solver.norm(u_exact)) / solver.norm(u_exact)

            norm_change.append(dnorm)

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
    plt.show()
