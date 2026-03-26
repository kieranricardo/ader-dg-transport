import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_3D.base_ader_dg_3D import BaseADERDG3D


class WaveAderDG3D(BaseADERDG3D):

    nvars = 4

    def __init__(self, xlim, ylim, zlim, nx, ny, nz, poly_order, c, dt):

        BaseADERDG3D.__init__(self, xlim, ylim, zlim, nx, ny, nz, poly_order)

        self.c = c

        self.dt = dt

        self.x_cfl = self.c * self.dt / self.dx
        self.y_cfl = self.c * self.dt / self.dy
        self.z_cfl = self.c * self.dt / self.dz

        n = self.poly_order + 1
        self.state = np.zeros((self.nx, self.ny, self.nz, self.nvars, n, n, n))
        self.u = self.state[:, :, :, 0]
        self.v = self.state[:, :, :, 1]
        self.w = self.state[:, :, :, 2]
        self.h = self.state[:, :, :, 3]

        self.time = 0.0

        (
            Dt, Dx, Dy, Dz, first_space_integral, last_space_integral,
            xm_integral, xp_integral, ym_integral, yp_integral, zm_integral, zp_integral
        ) = self.get_matrices()

        sz = n ** 4
        self.M1 = np.zeros((self.nvars * sz, self.nvars * sz))

        u_slice = slice(0, sz)
        v_slice = slice(sz, 2 * sz)
        w_slice = slice(2 * sz, 3 * sz)
        h_slice = slice(3 * sz, 4 * sz)

        self.M1[u_slice, u_slice] += Dt + first_space_integral
        self.M1[v_slice, v_slice] += Dt + first_space_integral
        self.M1[w_slice, w_slice] += Dt + first_space_integral
        self.M1[h_slice, h_slice] += Dt + first_space_integral

        # dudt + dhdx = 0
        self.M1[u_slice, h_slice] += self.x_cfl * Dx
        # dvdt + dhdy = 0
        self.M1[v_slice, h_slice] += self.y_cfl * Dy
        # dwdt + dhdz = 0
        self.M1[w_slice, h_slice] += self.z_cfl * Dz
        # dhdt + dudx + dvdy + dwdz = 0
        self.M1[h_slice, u_slice] += self.x_cfl * Dx
        self.M1[h_slice, v_slice] += self.y_cfl * Dy
        self.M1[h_slice, w_slice] += self.z_cfl * Dz

        self.M_nc = np.copy(self.M1)
        # dudt + dhdx = 0
        self.M_nc[u_slice, h_slice] += 0.5 * self.x_cfl * (xm_integral - xp_integral)
        # dvdt + dhdy = 0
        self.M_nc[v_slice, h_slice] += 0.5 * self.y_cfl * (ym_integral - yp_integral)
        # dwdt + dhdz = 0
        self.M_nc[w_slice, h_slice] += 0.5 * self.z_cfl * (zm_integral - zp_integral)
        # dhdt + dudx + dvdy + dwdz = 0
        self.M_nc[h_slice, u_slice] += 0.5 * self.x_cfl * (xm_integral - xp_integral)
        self.M_nc[h_slice, v_slice] += 0.5 * self.y_cfl * (ym_integral - yp_integral)
        self.M_nc[h_slice, w_slice] += 0.5 * self.z_cfl * (zm_integral - zp_integral)

        # dissipation terms
        # 0.5 * c * u * dy * dt
        self.M_nc[u_slice, u_slice] += 0.5 * self.x_cfl * (xm_integral + xp_integral)
        self.M_nc[v_slice, v_slice] += 0.5 * self.y_cfl * (ym_integral + yp_integral)
        self.M_nc[w_slice, w_slice] += 0.5 * self.z_cfl * (zm_integral + zp_integral)
        self.M_nc[h_slice, h_slice] += 0.5 * self.x_cfl * (xm_integral + xp_integral)
        self.M_nc[h_slice, h_slice] += 0.5 * self.y_cfl * (ym_integral + yp_integral)
        self.M_nc[h_slice, h_slice] += 0.5 * self.z_cfl * (zm_integral + zp_integral)

        self.M1_lu = lu_factor(self.M1)
        self.M_nc_lu = lu_factor(self.M_nc)


    def time_step(self, verbose=False, use_lu=True, multi_grid_pred=None, rhs_in=None, wt=1.0, tol=1e-8):

        if rhs_in is None:
            rhs_in = self.get_rhs(self.state)

        state_pred = self.preconditioner(rhs_in)

        for i in range(3):
            state_pred = self.block_jacobi(state_pred, rhs_in)

        self.state[:] = state_pred[:, :, :, :, -1]
        self.time += self.dt

        return 0

    def get_rhs(self, state):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, self.nz, self.nvars, n, n, n, n))

        for val, bdry_int in zip(self.get_vars(state), self.get_vars(bdry_integrals)):
            bdry_int[:, :, :, 0] += val / self.weights_x[-1]

        return bdry_integrals

    def get_vars(self, arr):
        return (arr[:, :, :, i] for i in range(self.nvars))

    def preconditioner(self, rhs_in):

        rhs = rhs_in.reshape(self.nx * self.ny * self.nz, -1).transpose()
        state_out = lu_solve(self.M1_lu, rhs).transpose().copy().reshape(rhs_in.shape)

        return state_out

    def _xbdry_nc(self, bdry_integrals, arr, xp, xm):

        u_bdry_integrals, _, _, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, _, _, h_pred = self.get_vars(arr)

        # x boundaries
        # flux fluc = 0.5 * (h_pred + h) - 0.5 * (u - u_pred) - h
        u_bdry_integrals[xp] += 0.5 * self.x_cfl * h_pred[xm] / self.weights_x[-1]
        u_bdry_integrals[xp] += 0.5 * self.x_cfl * u_pred[xm] / self.weights_x[-1]

        u_bdry_integrals[xm] -= 0.5 * self.x_cfl * h_pred[xp] / self.weights_x[-1]
        u_bdry_integrals[xm] += 0.5 * self.x_cfl * u_pred[xp] / self.weights_x[-1]

        h_bdry_integrals[xp] += 0.5 * self.x_cfl * u_pred[xm] / self.weights_x[-1]
        h_bdry_integrals[xp] += 0.5 * self.x_cfl * h_pred[xm] / self.weights_x[-1]

        h_bdry_integrals[xm] -= 0.5 * self.x_cfl * u_pred[xp] / self.weights_x[-1]
        h_bdry_integrals[xm] += 0.5 * self.x_cfl * h_pred[xp] / self.weights_x[-1]

    def _ybdry_nc(self, bdry_integrals, arr, yp, ym):

        _, v_bdry_integrals, _, h_bdry_integrals = self.get_vars(bdry_integrals)
        _, v_pred, _, h_pred = self.get_vars(arr)

        # y boundaries

        v_bdry_integrals[yp] += 0.5 * self.y_cfl * h_pred[ym] / self.weights_x[-1]
        v_bdry_integrals[yp] += 0.5 * self.y_cfl * v_pred[ym] / self.weights_x[-1]

        v_bdry_integrals[ym] -= 0.5 * self.y_cfl * h_pred[yp] / self.weights_x[-1]
        v_bdry_integrals[ym] += 0.5 * self.y_cfl * v_pred[yp] / self.weights_x[-1]

        h_bdry_integrals[yp] += 0.5 * self.y_cfl * v_pred[ym] / self.weights_x[-1]
        h_bdry_integrals[yp] += 0.5 * self.y_cfl * h_pred[ym] / self.weights_x[-1]

        h_bdry_integrals[ym] -= 0.5 * self.y_cfl * v_pred[yp] / self.weights_x[-1]
        h_bdry_integrals[ym] += 0.5 * self.y_cfl * h_pred[yp] / self.weights_x[-1]

    def _zbdry_nc(self, bdry_integrals, arr, yp, ym):

        _, _, w_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        _, _, w_pred, h_pred = self.get_vars(arr)

        # y boundaries

        w_bdry_integrals[yp] += 0.5 * self.z_cfl * h_pred[ym] / self.weights_x[-1]
        w_bdry_integrals[yp] += 0.5 * self.z_cfl * w_pred[ym] / self.weights_x[-1]

        w_bdry_integrals[ym] -= 0.5 * self.z_cfl * h_pred[yp] / self.weights_x[-1]
        w_bdry_integrals[ym] += 0.5 * self.z_cfl * w_pred[yp] / self.weights_x[-1]

        h_bdry_integrals[yp] += 0.5 * self.z_cfl * w_pred[ym] / self.weights_x[-1]
        h_bdry_integrals[yp] += 0.5 * self.z_cfl * h_pred[ym] / self.weights_x[-1]

        h_bdry_integrals[ym] -= 0.5 * self.z_cfl * w_pred[yp] / self.weights_x[-1]
        h_bdry_integrals[ym] += 0.5 * self.z_cfl * h_pred[yp] / self.weights_x[-1]

    def block_jacobi(self, state_in, rhs_in):
        bdry_integrals = np.copy(rhs_in)

        self._xbdry_nc(bdry_integrals, state_in, self.xp_int, self.xm_int)
        self._xbdry_nc(bdry_integrals, state_in, self.xp_ext, self.xm_ext)
        self._ybdry_nc(bdry_integrals, state_in, self.yp_int, self.ym_int)
        self._ybdry_nc(bdry_integrals, state_in, self.yp_ext, self.ym_ext)
        self._zbdry_nc(bdry_integrals, state_in, self.zp_int, self.zm_int)
        self._zbdry_nc(bdry_integrals, state_in, self.zp_ext, self.zm_ext)

        # rhs = bdry_integrals.reshape(self.nx * self.ny, -1)
        # state_out = np.einsum('ab,nb->na', self.M_nc_inv, rhs).reshape(rhs_in.shape)

        rhs = bdry_integrals.reshape(self.nx * self.ny * self.nz, -1).transpose()
        state_out = lu_solve(self.M_nc_lu, rhs).transpose().copy().reshape(bdry_integrals.shape)

        return state_out