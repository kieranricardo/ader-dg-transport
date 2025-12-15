import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D


class WaveStandardAderDG2D(BaseADERDG2D):

    def __init__(self, xlim, nx, poly_order, c, dt, f=0.0):

        ylim = xlim
        ny = nx

        BaseADERDG2D.__init__(self, xlim, ylim, nx, ny, poly_order)

        self.c = c
        self.f = f

        self.dt = dt

        self.x_cfl = self.c * self.dt / self.dx
        self.y_cfl = self.c * self.dt / self.dy

        self.ts = self.dt * self.taus

        self.time = 0.0

        self.a = 0.5

        n = self.poly_order + 1
        self.state = np.zeros((self.nx, self.ny, 3, n, n))
        self.u = self.state[:, :, 0]
        self.v = self.state[:, :, 1]
        self.h = self.state[:, :, 2]

        (Dt, Dx, Dy, volume_integral, first_space_integral,
         last_space_integral, xm_integral, xp_integral, ym_integral, yp_integral) = self.get_matrices()

        sz = n ** 3
        self.M1 = np.zeros((3 * n ** 3, 3 * n ** 3))

        u_slice = slice(0, sz)
        v_slice = slice(sz, 2 * sz)
        h_slice = slice(2 * sz, 3 * sz)

        self.M1[u_slice, u_slice] += Dt + first_space_integral
        self.M1[v_slice, v_slice] += Dt + first_space_integral
        self.M1[h_slice, h_slice] += Dt + first_space_integral

        self.M1[u_slice, v_slice] -= self.f * self.dt * np.eye(n ** 3)
        self.M1[v_slice, u_slice] += self.f * self.dt * np.eye(n ** 3)

        # dudt + dhdx = 0
        self.M1[u_slice, h_slice] += self.x_cfl * Dx
        # dvdt + dhdy = 0
        self.M1[v_slice, h_slice] += self.y_cfl * Dy
        # dhdt + dudx + dvdy = 0
        self.M1[h_slice, u_slice] += self.x_cfl * Dx
        self.M1[h_slice, v_slice] += self.y_cfl * Dy

        self.M1_lu = lu_factor(self.M1)

    def time_step(self):

        rhs_in = self.get_rhs(self.state)
        rhs = rhs_in.reshape(self.nx * self.ny, -1).transpose()
        state_pred = lu_solve(self.M1_lu, rhs).transpose().copy().reshape(rhs_in.shape)

        bdry_integrals = self.corrector(state_pred)
        diff = (bdry_integrals * self.weights_x[None, None, None, :, None, None]).sum(axis=3)

        self.state[:] += diff
        self.time += self.dt

        return 0

    def get_vars(self, arr):
        return (arr[:, :, i] for i in range(3))

    def get_rhs(self, state):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, 3, n, n, n))

        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u, v, h = self.get_vars(state)

        u_bdry_integrals[:, :, 0] += u / self.weights_x[-1]
        v_bdry_integrals[:, :, 0] += v / self.weights_x[-1]
        h_bdry_integrals[:, :, 0] += h / self.weights_x[-1]

        return bdry_integrals

    def _xbdry_corrector(self, bdry_integrals, arr, xp, xm):
        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, v_pred, h_pred = self.get_vars(arr)

        # x boundaries
        fluxp = h_pred[xp]
        fluxm = h_pred[xm]
        num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (u_pred[xp] - u_pred[xm])
        u_bdry_integrals[xp] += self.x_cfl * (num_flux - fluxp) / self.weights_x[-1]
        u_bdry_integrals[xm] -= self.x_cfl * (num_flux - fluxm) / self.weights_x[-1]

        fluxp = u_pred[xp]
        fluxm = u_pred[xm]
        num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (h_pred[xp] - h_pred[xm])
        h_bdry_integrals[xp] += self.x_cfl * (num_flux - fluxp) / self.weights_x[-1]
        h_bdry_integrals[xm] -= self.x_cfl * (num_flux - fluxm) / self.weights_x[-1]


    def _ybdry_corrector(self, bdry_integrals, arr, yp, ym):
        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, v_pred, h_pred = self.get_vars(arr)

        u_pred, v_pred, h_pred = self.get_vars(arr)
        # y boundaries
        fluxp = h_pred[yp]
        fluxm = h_pred[ym]
        num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (v_pred[yp] - v_pred[ym])
        v_bdry_integrals[yp] += self.y_cfl * (num_flux - fluxp) / self.weights_x[-1]
        v_bdry_integrals[ym] -= self.y_cfl * (num_flux - fluxm) / self.weights_x[-1]

        fluxp = v_pred[yp]
        fluxm = v_pred[ym]
        num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (h_pred[yp] - h_pred[ym])
        h_bdry_integrals[yp] += self.y_cfl * (num_flux - fluxp) / self.weights_x[-1]
        h_bdry_integrals[ym] -= self.y_cfl * (num_flux - fluxm) / self.weights_x[-1]

    def corrector(self, state_pred):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, 3, n, n, n))

        self._xbdry_corrector(bdry_integrals, state_pred, self.xp_int, self.xm_int)
        self._xbdry_corrector(bdry_integrals, state_pred, self.xp_ext, self.xm_ext)

        self._ybdry_corrector(bdry_integrals, state_pred, self.yp_int, self.ym_int)
        self._ybdry_corrector(bdry_integrals, state_pred, self.yp_ext, self.ym_ext)

        # volume terms
        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, v_pred, h_pred = self.get_vars(state_pred)

        u_bdry_integrals -= self.x_cfl * self.ddxi(h_pred)
        v_bdry_integrals -= self.y_cfl * self.ddeta(h_pred)
        h_bdry_integrals -= self.x_cfl * self.ddxi(u_pred) + self.y_cfl * self.ddeta(v_pred)

        return bdry_integrals

    def norm(self, u, v, h):
        return np.sqrt(self.integrate(u ** 2 + v ** 2 + h ** 2))