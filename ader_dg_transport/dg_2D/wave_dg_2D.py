import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.dg_2D.base_dg_2D import BaseDG2D


class WaveDG2D(BaseDG2D):

    def __init__(self, xlim, ylim, nx, ny, poly_order, c, dt, f=0.0):

        BaseDG2D.__init__(self, xlim, ylim, nx, ny, poly_order)


        self.c = c * np.ones((self.nx, self.ny, poly_order+1, poly_order+1))
        self.f = f

        self.dt = dt

        self.Jx = 0.5 * self.dx
        self.Jy = 0.5 * self.dy

        self.time = 0.0

        self.a = 0.5

        n = self.poly_order + 1
        self.state = np.zeros((3, self.nx, self.ny, n, n))
        self.u = self.state[:, :, 0]
        self.v = self.state[:, :, 1]
        self.h = self.state[:, :, 2]

        self.cpp = True

        self.y_periodic = True
        self.x_periodic = True

    def get_vars(self, arr):
        return (arr[i] for i in range(3))

    def norm(self, u, v, h):
        return np.sqrt(self.integrate(u ** 2 + v ** 2 + h ** 2))

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
        self.solve(u_tmp, dstatedt=k)
        i = 1
        if history_data is not None:
            history_data[i][:] = k / self.c
        if forcing is not None:
            k += forcing[i]
        if stage_data is not None:
            stage_data[i][:] = u_tmp

        u_tmp[:] = u_tmp[:] + 0.5 * dt * k
        self.solve(u_tmp, dstatedt=k)
        i = 2
        if history_data is not None:
            history_data[i][:] = k / self.c
        if forcing is not None:
            k += forcing[i]
        if stage_data is not None:
            stage_data[i][:] = u_tmp

        u_tmp[:] = (2 / 3) * self.state + (1 / 3) * u_tmp[:] + (1 / 6) * dt * k
        self.solve(u_tmp, dstatedt=k)
        i = 3
        if history_data is not None:
            history_data[i][:] = k / self.c
        if forcing is not None:
            k += forcing[i]
        if stage_data is not None:
            stage_data[i][:] = u_tmp

        self.state[:] = u_tmp + 0.5 * dt * k

        self.time += dt

    def solve(self, state, dstatedt):

        from ader_dg_transport import dg_wave_2D_volume_kernel

        u, v, h = self.get_vars(state)
        dstatedt[:] = 0.0
        dudt, dvdt, dhdt = self.get_vars(dstatedt)

        if self.cpp:
            dg_wave_2D_volume_kernel(u, v, h, dudt, dvdt, dhdt, self.c, self.D, self.Jx, self.Jy, self.weights_x[-1])
        else:
            dudt += -self.c * self.ddxi(h) / self.Jx
            dvdt += -self.c * self.ddeta(h) / self.Jy
            dhdt += -self.c * (self.ddxi(u) / self.Jx + self.ddeta(v) / self.Jy)

        def _interface_boundaries(ip, im, direction):

            state_p, dstatedt_p = self.get_boundary_data(state, ip), self.get_boundary_data(dstatedt, ip)
            state_m, dstatedt_m = self.get_boundary_data(state, im), self.get_boundary_data(dstatedt, im)
            cp = np.copy(self.c[ip])
            cm = np.copy(self.c[im])
            self.solve_boundaries(state_p, state_m, dstatedt_p, dstatedt_m, cp, cm, direction)
            dstatedt[(slice(None),) + ip] = dstatedt_p
            dstatedt[(slice(None),) + im] = dstatedt_m

        if not self.cpp:
            _interface_boundaries(self.xp_int, self.xm_int, 'x')
            _interface_boundaries(self.yp_int, self.ym_int, 'y')

        def _free_boundaries(ip, im, direction):

            state_p, dstatedt_p = self.get_boundary_data(state, ip), self.get_boundary_data(dstatedt, ip)
            dstatedt_alt = np.zeros_like(dstatedt_p)
            state_m = np.copy(state_p)
            um, vm, hm = self.get_vars(state_m)
            hm *= -1
            c_ = np.copy(self.c[ip])
            self.solve_boundaries(state_p, state_m, dstatedt_p, dstatedt_alt, c_, c_, direction)
            dstatedt[(slice(None),) + ip] = dstatedt_p

            state_m, dstatedt_m = self.get_boundary_data(state, im), self.get_boundary_data(dstatedt, im)
            state_p = np.copy(state_m)
            up, vp, hp = self.get_vars(state_p)
            hp *= -1
            c_ = np.copy(self.c[im])
            self.solve_boundaries(state_p, state_m, dstatedt_alt, dstatedt_m, c_, c_, direction)
            dstatedt[(slice(None),) + im] = dstatedt_m

        def _open_boundaries(ip, im, direction):

            assert direction == 'x'

            state_p, dstatedt_p = self.get_boundary_data(state, ip), self.get_boundary_data(dstatedt, ip)
            dstatedt_alt = np.zeros_like(dstatedt_p)
            state_m = np.copy(state_p)
            um, vm, hm = self.get_vars(state_m)
            # (um + hm) = 0.0 --> um = -hm
            # um - hm = (up - hp) ---> 2 * um = (up - hp)
            um[:] = 0.5 * (um - hm)
            hm[:] = -um
            c_ = np.copy(self.c[ip])
            self.solve_boundaries(state_p, state_m, dstatedt_p, dstatedt_alt, c_, c_, direction)
            dstatedt[(slice(None),) + ip] = dstatedt_p

            state_m, dstatedt_m = self.get_boundary_data(state, im), self.get_boundary_data(dstatedt, im)
            state_p = np.copy(state_m)
            up, vp, hp = self.get_vars(state_p)
            # (up - hp) = 0.0 --> um = hm
            # um + hm = (up + hp) ---> 2 * um = (up + hp)
            up[:] = 0.5 * (up + hp)
            hp[:] = up
            c_ = np.copy(self.c[im])
            self.solve_boundaries(state_p, state_m, dstatedt_alt, dstatedt_m, c_, c_, direction)
            dstatedt[(slice(None),) + im] = dstatedt_m

        if self.x_periodic:
            _interface_boundaries(self.xp_ext, self.xm_ext, 'x')
        else:
            _open_boundaries(self.xp_ext, self.xm_ext, 'x')

        if self.y_periodic:
            _interface_boundaries(self.yp_ext, self.ym_ext, 'y')
        else:
            _free_boundaries(self.yp_ext, self.ym_ext, 'y')

    def get_boundary_data(self, state, idx):
        # extract boundary data
        state_bdry = np.copy(state[(slice(None),) + idx])
        return state_bdry

    def solve_boundaries(self, state_p, state_m, dstatedt_p, dstatedt_m, cp, cm, direction):

        from ader_dg_transport import dg_wave_2D_bdry_kernel
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
            shape = (-1, self.poly_order + 1)
            dg_wave_2D_bdry_kernel(
                up.reshape(shape), hp.reshape(shape), um.reshape(shape), hm.reshape(shape),
                dudtp.reshape(shape), dhdtp.reshape(shape), dudtm.reshape(shape), dhdtm.reshape(shape),
                cp.reshape(shape), cm.reshape(shape), J, self.weights_x[-1]
            )
        else:
            fluxp = hp
            fluxm = hm
            num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (up - um)
            dudtp += (num_flux - fluxp) * cp / (self.weights_x[-1] * self.Jx)
            dudtm += -(num_flux - fluxm) * cm / (self.weights_x[-1] * self.Jx)

            fluxp = up
            fluxm = um
            num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (hp - hm)
            dhdtp += (num_flux - fluxp) * cp / (self.weights_x[-1] * self.Jy)
            dhdtm += -(num_flux - fluxm) * cm / (self.weights_x[-1] * self.Jy)

        return 0.0

