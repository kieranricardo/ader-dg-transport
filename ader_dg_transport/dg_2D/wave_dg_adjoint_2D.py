import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.dg_2D.wave_dg_2D import WaveDG2D


class WaveAdjointDG2D(WaveDG2D):

    def __init__(self, xlim, ylim, nx, ny, poly_order, c, dt, f=0.0):

        WaveDG2D.__init__(self, xlim, ylim, nx, ny, poly_order, c, dt, f)

    def solve(self, state, dstatedt):

        u, v, h = self.get_vars(state)
        dstatedt[:] = 0.0
        dudt, dvdt, dhdt = self.get_vars(dstatedt)

        if self.cpp:
            from ader_dg_transport import dg_wave_adjoint_2D_volume_kernel
            dg_wave_adjoint_2D_volume_kernel(u, v, h, dudt, dvdt, dhdt, self.c, self.D, self.Jx, self.Jy, self.weights_x[-1])
        else:
            dudt += self.ddxi(self.c * h) / self.Jx
            dvdt += self.ddeta(self.c * h) / self.Jy
            dhdt += (self.ddxi(self.c * u) / self.Jx + self.ddeta(self.c * v) / self.Jy)

        bdry_list = ((self.xp_int, self.xm_int, 'x'), (self.xp_ext, self.xm_ext, 'x'), (self.yp_int, self.ym_int, 'y'), (self.yp_ext, self.ym_ext, 'y'))

        for (ip, im, direction) in bdry_list:

            state_p, dstatedt_p = self.get_boundary_data(state, ip), self.get_boundary_data(dstatedt, ip)
            state_m, dstatedt_m = self.get_boundary_data(state, im), self.get_boundary_data(dstatedt, im)
            cp = np.copy(self.c[ip])
            cm = np.copy(self.c[im])
            self.solve_boundaries(state_p, state_m, dstatedt_p, dstatedt_m, cp, cm, direction)
            dstatedt[(slice(None),) + ip] = dstatedt_p
            dstatedt[(slice(None),) + im] = dstatedt_m

    def solve_boundaries(self, state_p, state_m, dstatedt_p, dstatedt_m, cp, cm, direction):

        if direction == 'x':
            up, _, hp = self.get_vars(state_p)
            um, _, hm = self.get_vars(state_m)

            dudtp, _, dhdtp = self.get_vars(dstatedt_p)
            dudtm, _, dhdtm = self.get_vars(dstatedt_m)
            J = self.Jx
        else:
            _, up, hp = self.get_vars(state_p)
            _, um, hm = self.get_vars(state_m)

            _, dudtp, dhdtp = self.get_vars(dstatedt_p)
            _, dudtm, dhdtm = self.get_vars(dstatedt_m)
            J = self.Jy

        if self.cpp:
            from ader_dg_transport import dg_wave_adjoint_2D_bdry_kernel
            shape = (-1, self.poly_order + 1)
            dg_wave_adjoint_2D_bdry_kernel(
                up.reshape(shape), hp.reshape(shape), um.reshape(shape), hm.reshape(shape),
                dudtp.reshape(shape), dhdtp.reshape(shape), dudtm.reshape(shape), dhdtm.reshape(shape),
                cp.reshape(shape), cm.reshape(shape), J, self.weights_x[-1]
            )
        else:
            fluxp = -cp * hp
            fluxm = -cm * hm
            num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (cp * up - cm * um)
            dudtp += (num_flux - fluxp) / (self.weights_x[-1] * self.Jx)
            dudtm += -(num_flux - fluxm) / (self.weights_x[-1] * self.Jx)

            fluxp = -cp * up
            fluxm = -cm * um
            num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (cp * hp - cm * hm)
            dhdtp += (num_flux - fluxp) / (self.weights_x[-1] * self.Jy)
            dhdtm += -(num_flux - fluxm) / (self.weights_x[-1] * self.Jy)

        return 0.0

    def time_step(self, dt=None, forcing=None, history_data=None, stage_data=None):

        if dt is None:
            dt = self.dt

        k = np.zeros_like(self.state)
        u_tmp = np.zeros_like(self.state)

        self.solve(self.state, dstatedt=k)
        i = 0
        if history_data is not None:
            history_data[i][:] = k / self.c
        if forcing is not None:
            k += forcing[i]
        if stage_data is not None:
            stage_data[i][:] = self.state

        u_tmp[:] = self.state + 0.5 * dt * k
        lambda_3 = np.copy(u_tmp)
        self.solve(u_tmp, dstatedt=k)
        i = 1
        if history_data is not None:
            history_data[i][:] = k / self.c
        if forcing is not None:
            k += forcing[i]
        if stage_data is not None:
            stage_data[i][:] = u_tmp

        u_tmp[:] = (u_tmp[:] + 0.5 * dt * k) / 3
        self.solve(u_tmp, dstatedt=k)
        i = 2
        if history_data is not None:
            history_data[i][:] = k / self.c
        if forcing is not None:
            k += forcing[i]
        if stage_data is not None:
            stage_data[i][:] = u_tmp

        u_tmp[:] = u_tmp[:] + 0.5 * dt * k
        self.solve(u_tmp, dstatedt=k)
        i = 3
        if history_data is not None:
            history_data[i][:] = k / self.c
        if forcing is not None:
            k += forcing[i]
        if stage_data is not None:
            stage_data[i][:] = u_tmp

        self.state[:] = u_tmp + 0.5 * dt * k + (2 / 3) * lambda_3
        self.time += dt
