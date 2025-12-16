import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D


class ElasticWaveAderDG2D(BaseADERDG2D):

    nvars = 5

    def __init__(self, xlim, nx, poly_order, rho, L, mu, dt):

        ylim = xlim
        ny = nx

        BaseADERDG2D.__init__(self, xlim, ylim, nx, ny, poly_order)

        self.L = L
        self.mu = mu
        self.Lmu = self.L + 2 * self.mu
        self.rho = rho
        self.cp = np.sqrt(self.Lmu / self.rho)
        self.cs = np.sqrt(mu / self.rho)

        self.C_mat = np.array([
            self.Lmu, self.L, 0.0,
             self.L, self.Lmu, 0.0,
             0.0, 0.0, self.mu
            ]).reshape((3, 3))
        print(self.C_mat.shape)
        self.S_mat = np.linalg.inv(self.C_mat)

        self.dt = dt

        self.x_cfl = self.dt / self.dx
        self.y_cfl = self.dt / self.dy

        self.ts = self.dt * self.taus

        self.time = 0.0

        self.a = 0.5

        n = self.poly_order + 1
        self.state = np.zeros((self.nx, self.ny, self.nvars, n, n))
        self.u = self.state[:, :, 0]
        self.v = self.state[:, :, 1]
        self.oxx = self.state[:, :, 2]
        self.oyy = self.state[:, :, 3]
        self.oxy = self.state[:, :, 4]

        (Dt, Dx, Dy, volume_integral, first_space_integral,
         last_space_integral, xm_integral, xp_integral, ym_integral, yp_integral) = self.get_matrices()

        sz = n ** 3
        self.M1 = np.zeros((self.nvars * n ** 3, self.nvars * n ** 3))

        u_slice = slice(0, sz)
        v_slice = slice(sz, 2 * sz)
        oxx_slice = slice(2 * sz, 3 * sz)
        oyy_slice = slice(3 * sz, 4 * sz)
        oxy_slice = slice(4 * sz, 5 * sz)

        self.M1[u_slice, u_slice] += Dt + first_space_integral
        self.M1[v_slice, v_slice] += Dt + first_space_integral
        self.M1[oxx_slice, oxx_slice] += Dt + first_space_integral
        self.M1[oyy_slice, oyy_slice] += Dt + first_space_integral
        self.M1[oxy_slice, oxy_slice] += Dt + first_space_integral

        # dudt
        self.M1[u_slice, oxx_slice] += -(1 / self.rho) * self.x_cfl * Dx
        self.M1[u_slice, oxy_slice] += -(1 / self.rho) * self.y_cfl * Dy
        # dvdt
        self.M1[v_slice, oyy_slice] += -(1 / self.rho) * self.y_cfl * Dy
        self.M1[v_slice, oxy_slice] += -(1 / self.rho) * self.x_cfl * Dx
        # doxxdt
        self.M1[oxx_slice, u_slice] += -(2 * mu + L) * self.x_cfl * Dx
        self.M1[oxx_slice, v_slice] += -L * self.y_cfl * Dy
        # doyydt
        self.M1[oyy_slice, u_slice] +=  -L * self.x_cfl * Dx
        self.M1[oyy_slice, v_slice] += -(2 * mu + L) * self.y_cfl * Dy
        # doxydt
        self.M1[oxy_slice, u_slice] += -mu * self.y_cfl * Dy
        self.M1[oxy_slice, v_slice] += -mu * self.x_cfl * Dx

        self.M_nc = np.copy(self.M1)

        # x bdry fluxes
        ## centred part
        self.M_nc[u_slice, oxx_slice] += 0.5 * self.x_cfl * (xm_integral - xp_integral) * (-1 / self.rho)
        self.M_nc[oxx_slice, u_slice] += 0.5 * self.x_cfl * (xm_integral - xp_integral) * (-self.Lmu)
        self.M_nc[oyy_slice, u_slice] += 0.5 * self.x_cfl * (xm_integral - xp_integral) * (-self.L)
        self.M_nc[v_slice, oxy_slice] += 0.5 * self.x_cfl * (xm_integral - xp_integral) * (-1 / self.rho)
        self.M_nc[oxy_slice, v_slice] += 0.5 * self.x_cfl * (xm_integral - xp_integral) * (-self.mu)
        # dissipative part
        self.M_nc[u_slice, u_slice] += 0.5 * self.x_cfl * (xm_integral + xp_integral) * self.cp
        self.M_nc[oxx_slice, oxx_slice] += 0.5 * self.x_cfl * (xm_integral + xp_integral) * self.cp
        self.M_nc[oyy_slice, oxx_slice] += 0.5 * self.x_cfl * (xm_integral + xp_integral) * (self.L * (self.cp / self.Lmu))
        self.M_nc[v_slice, v_slice] += 0.5 * self.x_cfl * (xm_integral + xp_integral) * self.cs
        self.M_nc[oxy_slice, oxy_slice] += 0.5 * self.x_cfl * (xm_integral + xp_integral) * self.cs

        # y bdry fluxes
        ## centred part
        self.M_nc[v_slice, oyy_slice] += 0.5 * self.y_cfl * (ym_integral - yp_integral) * (-1 / self.rho)
        self.M_nc[oyy_slice, v_slice] += 0.5 * self.y_cfl * (ym_integral - yp_integral) * (-self.Lmu)
        self.M_nc[oxx_slice, v_slice] += 0.5 * self.y_cfl * (ym_integral - yp_integral) * (-self.L)
        self.M_nc[u_slice, oxy_slice] += 0.5 * self.y_cfl * (ym_integral - yp_integral) * (-1 / self.rho)
        self.M_nc[oxy_slice, u_slice] += 0.5 * self.y_cfl * (ym_integral - yp_integral) * (-self.mu)
        # dissipative part
        self.M_nc[v_slice, v_slice] += 0.5 * self.y_cfl * (ym_integral + yp_integral) * self.cp
        self.M_nc[oyy_slice, oyy_slice] += 0.5 * self.y_cfl * (ym_integral + yp_integral) * self.cp
        self.M_nc[oxx_slice, oyy_slice] += 0.5 * self.y_cfl * (ym_integral + yp_integral) * (self.L * (self.cp / self.Lmu))
        self.M_nc[u_slice, u_slice] += 0.5 * self.y_cfl * (ym_integral + yp_integral) * self.cs
        self.M_nc[oxy_slice, oxy_slice] += 0.5 * self.y_cfl * (ym_integral + yp_integral) * self.cs

        self.M1_lu = lu_factor(self.M1)
        self.M_nc_lu = lu_factor(self.M_nc)

    def time_step(self):

        rhs_in = self.get_rhs(self.state)
        state_pred = self.preconditioner(rhs_in)

        for _ in range(3):
            state_pred = self.block_jacobi(state_pred, rhs_in)

        self.state[:] = state_pred[:, :, :, -1]
        self.time += self.dt

        return 0

    def get_vars(self, arr):
        return (arr[:, :, i] for i in range(self.nvars))

    def get_rhs(self, state):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, self.nvars, n, n, n))

        for val, bdry_int in zip(self.get_vars(state), self.get_vars(bdry_integrals)):
            bdry_int[:, :, 0] += val / self.weights_x[-1]

        return bdry_integrals

    def block_jacobi(self, state_in, rhs_in):
        bdry_integrals = np.copy(rhs_in)

        self._xbdry_nc(bdry_integrals, state_in, self.xp_int, self.xm_int)
        self._xbdry_nc(bdry_integrals, state_in, self.xp_ext, self.xm_ext)
        self._ybdry_nc(bdry_integrals, state_in, self.yp_int, self.ym_int)
        self._ybdry_nc(bdry_integrals, state_in, self.yp_ext, self.ym_ext)

        # rhs = bdry_integrals.reshape(self.nx * self.ny, -1)
        # state_out = np.einsum('ab,nb->na', self.M_nc_inv, rhs).reshape(rhs_in.shape)

        rhs = bdry_integrals.reshape(self.nx * self.ny, -1).transpose()
        state_out = lu_solve(self.M_nc_lu, rhs).transpose().copy().reshape(bdry_integrals.shape)

        return state_out

    def preconditioner(self, rhs_in):

        rhs = rhs_in.reshape(self.nx * self.ny, -1).transpose()
        state_out = lu_solve(self.M1_lu, rhs).transpose().copy().reshape(rhs_in.shape)

        return state_out

    def _xbdry_nc(self, bdry_integrals, arr, ip, im):

        u_bdry, v_bdry, oxx_bdry, oyy_bdry, oxy_bdry = self.get_vars(bdry_integrals)
        u, v, oxx, oyy, oxy = self.get_vars(arr)

        # x boundaries
        # p characteristics
        u_bdry[ip] += self.x_cfl * (-0.5 * oxx[im] / self.rho) / self.weights_x[-1]
        u_bdry[ip] += self.x_cfl * (0.5 * self.cp * u[im]) / self.weights_x[-1]

        u_bdry[im] -= self.x_cfl * (-0.5 * oxx[ip] / self.rho) / self.weights_x[-1]
        u_bdry[im] += self.x_cfl * (0.5 * self.cp * u[ip]) / self.weights_x[-1]

        oxx_bdry[ip] += self.x_cfl * (-0.5 * self.Lmu * u[im]) / self.weights_x[-1]
        oxx_bdry[ip] += self.x_cfl * (0.5 * self.cp * oxx[im]) / self.weights_x[-1]

        oxx_bdry[im] -= self.x_cfl * (-0.5 * self.Lmu * u[ip]) / self.weights_x[-1]
        oxx_bdry[im] += self.x_cfl * (0.5 * self.cp * oxx[ip]) / self.weights_x[-1]

        oyy_bdry[ip] += self.x_cfl * (-0.5 * self.L * u[im]) / self.weights_x[-1]
        oyy_bdry[ip] += self.x_cfl * (0.5 * self.L * (self.cp / self.Lmu) * oxx[im]) / self.weights_x[-1]

        oyy_bdry[im] -= self.x_cfl * (-0.5 * self.L * u[ip]) / self.weights_x[-1]
        oyy_bdry[im] += self.x_cfl * (0.5 * self.L * (self.cp / self.Lmu) * oxx[ip]) / self.weights_x[-1]

        # s characteristics
        v_bdry[ip] += self.x_cfl * (-0.5 * oxy[im] / self.rho) / self.weights_x[-1]
        v_bdry[ip] += self.x_cfl * (0.5 * self.cs * v[im]) / self.weights_x[-1]

        v_bdry[im] -= self.x_cfl * (-0.5 * oxy[ip] / self.rho) / self.weights_x[-1]
        v_bdry[im] += self.x_cfl * (0.5 * self.cs * v[ip]) / self.weights_x[-1]

        oxy_bdry[ip] += self.x_cfl * (-0.5 * self.mu * v[im]) / self.weights_x[-1]
        oxy_bdry[ip] += self.x_cfl * (0.5 * self.cs * oxy[im]) / self.weights_x[-1]

        oxy_bdry[im] -= self.x_cfl * (-0.5 * self.mu * v[ip]) / self.weights_x[-1]
        oxy_bdry[im] += self.x_cfl * (0.5 * self.cs * oxy[ip]) / self.weights_x[-1]

    def _ybdry_nc(self, bdry_integrals, arr, ip, im):

        u_bdry, v_bdry, oxx_bdry, oyy_bdry, oxy_bdry = self.get_vars(bdry_integrals)
        u, v, oxx, oyy, oxy = self.get_vars(arr)

        # x boundaries
        # p characteristics
        v_bdry[ip] += self.y_cfl * (-0.5 * oyy[im] / self.rho) / self.weights_x[-1]
        v_bdry[ip] += self.y_cfl * (0.5 * self.cp * v[im]) / self.weights_x[-1]

        v_bdry[im] -= self.y_cfl * (-0.5 * oyy[ip] / self.rho) / self.weights_x[-1]
        v_bdry[im] += self.y_cfl * (0.5 * self.cp * v[ip]) / self.weights_x[-1]

        oyy_bdry[ip] += self.y_cfl * (-0.5 * self.Lmu * v[im]) / self.weights_x[-1]
        oyy_bdry[ip] += self.y_cfl * (0.5 * self.cp * oyy[im]) / self.weights_x[-1]

        oyy_bdry[im] -= self.y_cfl * (-0.5 * self.Lmu * v[ip]) / self.weights_x[-1]
        oyy_bdry[im] += self.y_cfl * (0.5 * self.cp * oyy[ip]) / self.weights_x[-1]

        oxx_bdry[ip] += self.y_cfl * (-0.5 * self.L * v[im]) / self.weights_x[-1]
        oxx_bdry[ip] += self.y_cfl * (0.5 * self.L * (self.cp / self.Lmu) * oyy[im]) / self.weights_x[-1]

        oxx_bdry[im] -= self.y_cfl * (-0.5 * self.L * v[ip]) / self.weights_x[-1]
        oxx_bdry[im] += self.y_cfl * (0.5 * self.L * (self.cp / self.Lmu) * oyy[ip]) / self.weights_x[-1]

        # s characteristics
        u_bdry[ip] += self.y_cfl * (-0.5 * oxy[im] / self.rho) / self.weights_x[-1]
        u_bdry[ip] += self.y_cfl * (0.5 * self.cs * u[im]) / self.weights_x[-1]

        u_bdry[im] -= self.y_cfl * (-0.5 * oxy[ip] / self.rho) / self.weights_x[-1]
        u_bdry[im] += self.y_cfl * (0.5 * self.cs * u[ip]) / self.weights_x[-1]

        oxy_bdry[ip] += self.y_cfl * (-0.5 * self.mu * u[im]) / self.weights_x[-1]
        oxy_bdry[ip] += self.y_cfl * (0.5 * self.cs * oxy[im]) / self.weights_x[-1]

        oxy_bdry[im] -= self.y_cfl * (-0.5 * self.mu * u[ip]) / self.weights_x[-1]
        oxy_bdry[im] += self.y_cfl * (0.5 * self.cs * oxy[ip]) / self.weights_x[-1]

    def norm(self, u, v, oxx, oyy, oxy):

        KE = 0.5 * self.rho * (u**2 + v**2)
        PE = 0.5 * oxx * (self.S_mat[0, 0] * oxx + self.S_mat[0, 1] * oyy)
        PE += 0.5 * oyy * (self.S_mat[1, 0] * oxx + self.S_mat[1, 1] * oyy)
        PE += 0.5 * oxy * self.S_mat[2, 2] * oxy
        return np.sqrt(self.integrate(KE + PE))