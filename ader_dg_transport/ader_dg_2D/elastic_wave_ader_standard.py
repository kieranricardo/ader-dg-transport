import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D


class ElasticWaveStandardAderDG2D(BaseADERDG2D):

    nvars = 5

    def __init__(self, xlim, nx, poly_order, rho, L, mu, dt):

        ylim = xlim
        ny = nx

        BaseADERDG2D.__init__(self, xlim, ylim, nx, ny, poly_order)

        self.L = L
        self.mu = mu
        self.rho = rho
        self.cp = np.sqrt((2 * self.mu + self.L) / self.rho)
        self.cs = np.sqrt(mu / self.rho)

        self.C_mat = np.array([
            self.L + 2 * self.mu, self.L, 0.0,
             self.L, self.L + 2 * self.mu, 0.0,
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
        return (arr[:, :, i] for i in range(self.nvars))

    def get_rhs(self, state):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, self.nvars, n, n, n))

        for val, bdry_int in zip(self.get_vars(state), self.get_vars(bdry_integrals)):
            bdry_int[:, :, 0] += val / self.weights_x[-1]

        return bdry_integrals

    def _xbdry_corrector(self, bdry_integrals, arr, ip, im):
        u_bdry, v_bdry, oxx_bdry, oyy_bdry, oxy_bdry = self.get_vars(bdry_integrals)
        u, v, oxx, oyy, oxy = self.get_vars(arr)

        u_hat = 0.5 * (u[ip] + u[im]) + 0.5 * (self.cp / (self.L + 2 * self.mu)) * (oxx[ip] - oxx[im])
        oxx_hat = 0.5 * (oxx[ip] + oxx[im]) + 0.5 * self.rho * self.cp * (u[ip] - u[im])

        v_hat = 0.5 * (v[ip] + v[im]) + 0.5 * (self.cs / self.mu) * (oxy[ip] - oxy[im])
        oxy_hat = 0.5 * (oxy[ip] + oxy[im]) + 0.5 * self.rho * self.cs * (v[ip] - v[im])

        # x boundaries
        # s characteristics
        fluxp = -(1 / self.rho) * oxx[ip]
        fluxm = -(1 / self.rho) * oxx[im]
        num_flux = -(1 / self.rho) * oxx_hat
        u_bdry[ip] += self.x_cfl * (num_flux - fluxp) / self.weights_x[-1]
        u_bdry[im] -= self.x_cfl * (num_flux - fluxm) / self.weights_x[-1]

        fluxp = -(self.L + 2 * self.mu) * u[ip]
        fluxm = -(self.L + 2 * self.mu) * u[im]
        num_flux = -(self.L + 2 * self.mu) * u_hat
        oxx_bdry[ip] += self.x_cfl * (num_flux - fluxp) / self.weights_x[-1]
        oxx_bdry[im] -= self.x_cfl * (num_flux - fluxm) / self.weights_x[-1]

        fluxp = -self.L * u[ip]
        fluxm = -self.L * u[im]
        num_flux = -self.L * u_hat
        oyy_bdry[ip] += self.x_cfl * (num_flux - fluxp) / self.weights_x[-1]
        oyy_bdry[im] -= self.x_cfl * (num_flux - fluxm) / self.weights_x[-1]

        # p characteristics
        fluxp = -(1 / self.rho) * oxy[ip]
        fluxm = -(1 / self.rho) * oxy[im]
        num_flux = -(1 / self.rho) * oxy_hat
        v_bdry[ip] += self.x_cfl * (num_flux - fluxp) / self.weights_x[-1]
        v_bdry[im] -= self.x_cfl * (num_flux - fluxm) / self.weights_x[-1]

        fluxp = -self.mu * v[ip]
        fluxm = -self.mu * v[im]
        num_flux = -self.mu * v_hat
        oxy_bdry[ip] += self.x_cfl * (num_flux - fluxp) / self.weights_x[-1]
        oxy_bdry[im] -= self.x_cfl * (num_flux - fluxm) / self.weights_x[-1]


    def _ybdry_corrector(self, bdry_integrals, arr, ip, im):

        u_bdry, v_bdry, oxx_bdry, oyy_bdry, oxy_bdry = self.get_vars(bdry_integrals)
        u, v, oxx, oyy, oxy = self.get_vars(arr)

        v_hat = 0.5 * (v[ip] + v[im]) + 0.5 * (self.cp / (self.L + 2 * self.mu)) * (oyy[ip] - oyy[im])
        oyy_hat = 0.5 * (oyy[ip] + oyy[im]) + 0.5 * self.rho * self.cp * (v[ip] - v[im])

        u_hat = 0.5 * (u[ip] + u[im]) + 0.5 * (self.cs / self.mu) * (oxy[ip] - oxy[im])
        oxy_hat = 0.5 * (oxy[ip] + oxy[im]) + 0.5 * self.rho * self.cs * (u[ip] - u[im])

        # y boundaries
        # s characteristics
        fluxp = -(1 / self.rho) * oyy[ip]
        fluxm = -(1 / self.rho) * oyy[im]
        num_flux = -(1 / self.rho) * oyy_hat
        v_bdry[ip] += self.y_cfl * (num_flux - fluxp) / self.weights_x[-1]
        v_bdry[im] -= self.y_cfl * (num_flux - fluxm) / self.weights_x[-1]

        fluxp = -(self.L + 2 * self.mu) * v[ip]
        fluxm = -(self.L + 2 * self.mu) * v[im]
        num_flux = -(self.L + 2 * self.mu) * v_hat
        oyy_bdry[ip] += self.y_cfl * (num_flux - fluxp) / self.weights_x[-1]
        oyy_bdry[im] -= self.y_cfl * (num_flux - fluxm) / self.weights_x[-1]

        fluxp = -self.L * v[ip]
        fluxm = -self.L * v[im]
        num_flux = -self.L * v_hat
        oxx_bdry[ip] += self.y_cfl * (num_flux - fluxp) / self.weights_x[-1]
        oxx_bdry[im] -= self.y_cfl * (num_flux - fluxm) / self.weights_x[-1]

        # p characteristics
        fluxp = -(1 / self.rho) * oxy[ip]
        fluxm = -(1 / self.rho) * oxy[im]
        num_flux = -(1 / self.rho) * oxy_hat
        u_bdry[ip] += self.y_cfl * (num_flux - fluxp) / self.weights_x[-1]
        u_bdry[im] -= self.y_cfl * (num_flux - fluxm) / self.weights_x[-1]

        fluxp = -self.mu * u[ip]
        fluxm = -self.mu * u[im]
        num_flux = -self.mu * u_hat
        oxy_bdry[ip] += self.y_cfl * (num_flux - fluxp) / self.weights_x[-1]
        oxy_bdry[im] -= self.y_cfl * (num_flux - fluxm) / self.weights_x[-1]

    def corrector(self, state_pred):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, self.nvars, n, n, n))

        self._xbdry_corrector(bdry_integrals, state_pred, self.xp_int, self.xm_int)
        self._xbdry_corrector(bdry_integrals, state_pred, self.xp_ext, self.xm_ext)

        self._ybdry_corrector(bdry_integrals, state_pred, self.yp_int, self.ym_int)
        self._ybdry_corrector(bdry_integrals, state_pred, self.yp_ext, self.ym_ext)

        # volume terms
        u_bdry, v_bdry, oxx_bdry, oyy_bdry, oxy_bdry = self.get_vars(bdry_integrals)
        u, v, oxx, oyy, oxy = self.get_vars(state_pred)

        u_bdry += (self.x_cfl / self.rho) * self.ddxi(oxx) + (self.y_cfl / self.rho) * self.ddeta(oxy)
        v_bdry += (self.x_cfl / self.rho) * self.ddxi(oxy) + (self.y_cfl / self.rho) * self.ddeta(oyy)

        oxx_bdry += (self.L + 2 * self.mu) * self.x_cfl * self.ddxi(u) + self.L * self.y_cfl * self.ddeta(v)
        oyy_bdry += self.L * self.x_cfl * self.ddxi(u) + (self.L + 2 * self.mu) * self.y_cfl * self.ddeta(v)

        oxy_bdry += self.mu * self.x_cfl * self.ddxi(v) + self.mu * self.y_cfl * self.ddeta(u)

        return bdry_integrals

    def norm(self, u, v, oxx, oyy, oxy):

        KE = 0.5 * self.rho * (u**2 + v**2)
        PE = 0.5 * oxx * (self.S_mat[0, 0] * oxx + self.S_mat[0, 1] * oyy)
        PE += 0.5 * oyy * (self.S_mat[1, 0] * oxx + self.S_mat[1, 1] * oyy)
        PE += 0.5 * oxy * self.S_mat[2, 2] * oxy
        return np.sqrt(self.integrate(KE + PE))