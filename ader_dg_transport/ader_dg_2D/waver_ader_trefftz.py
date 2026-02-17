import numpy as np
from scipy.linalg import lu_factor, lu_solve, null_space
from scipy.interpolate import lagrange
import scipy
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D


class WaveAderTrefftzDG2D(BaseADERDG2D):

    def __init__(self, xlim, ylim, nx, ny, poly_order, c, dt):
        BaseADERDG2D.__init__(self, xlim, ylim, nx, ny, poly_order)

        self.c = c
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

        u_slice = slice(0, sz)
        v_slice = slice(sz, 2 * sz)
        h_slice = slice(2 * sz, 3 * sz)

        # make basis functions
        self.M0 = np.zeros((3 * n ** 3, 3 * n ** 3))
        self.M0[u_slice, u_slice] += Dt
        self.M0[v_slice, v_slice] += Dt
        self.M0[h_slice, h_slice] += Dt
        # dudt + dhdx = 0
        self.M0[u_slice, h_slice] += self.x_cfl * Dx
        # dvdt + dhdy = 0
        self.M0[v_slice, h_slice] += self.y_cfl * Dy
        # dhdt + dudx + dvdy = 0
        self.M0[h_slice, u_slice] += self.x_cfl * Dx
        self.M0[h_slice, v_slice] += self.y_cfl * Dy

        self.basis_functions = null_space(self.M0).transpose().reshape(-1, 3, n, n, n)
        self.basis_functions_weighted = self.basis_functions * (self.weights_2D[..., None] * self.weights_x[None, None, :])[None, None, :]

        # matrices

        u_basis_functions = self.basis_functions[:, 0, 0].reshape(-1, n * n)
        v_basis_functions = self.basis_functions[:, 1, 0].reshape(-1, n * n)
        h_basis_functions = self.basis_functions[:, 2, 0].reshape(-1, n * n)

        ic_mm = u_basis_functions @ np.diag(self.weights_2D.ravel()) @ u_basis_functions.T
        ic_mm += v_basis_functions @ np.diag(self.weights_2D.ravel()) @ v_basis_functions.T
        ic_mm += h_basis_functions @ np.diag(self.weights_2D.ravel()) @ h_basis_functions.T

        # horizontal boundaries
        u_basis_functions = self.basis_functions[:, 0, :, -1].reshape(-1, n * n)
        h_basis_functions = self.basis_functions[:, 2, :, -1].reshape(-1, n * n)

        xp_mm = u_basis_functions @ np.diag(self.weights_2D.ravel()) @ h_basis_functions.T
        xp_mm += h_basis_functions @ np.diag(self.weights_2D.ravel()) @ u_basis_functions.T
        xp_mm += -u_basis_functions @ np.diag(self.weights_2D.ravel()) @ u_basis_functions.T
        xp_mm += -h_basis_functions @ np.diag(self.weights_2D.ravel()) @ h_basis_functions.T

        u_basis_functions = self.basis_functions[:, 0, :, 0].reshape(-1, n * n)
        h_basis_functions = self.basis_functions[:, 2, :, 0].reshape(-1, n * n)

        xm_mm = u_basis_functions @ np.diag(self.weights_2D.ravel()) @ h_basis_functions.T
        xm_mm += h_basis_functions @ np.diag(self.weights_2D.ravel()) @ u_basis_functions.T
        xm_mm += u_basis_functions @ np.diag(self.weights_2D.ravel()) @ u_basis_functions.T
        xm_mm += h_basis_functions @ np.diag(self.weights_2D.ravel()) @ h_basis_functions.T

        # vertical boundaries
        v_basis_functions = self.basis_functions[:, 1, :, :, -1].reshape(-1, n * n)
        h_basis_functions = self.basis_functions[:, 2, :, :, -1].reshape(-1, n * n)

        yp_mm = v_basis_functions @ np.diag(self.weights_2D.ravel()) @ h_basis_functions.T
        yp_mm += h_basis_functions @ np.diag(self.weights_2D.ravel()) @ v_basis_functions.T
        yp_mm += -v_basis_functions @ np.diag(self.weights_2D.ravel()) @ v_basis_functions.T
        yp_mm += -h_basis_functions @ np.diag(self.weights_2D.ravel()) @ h_basis_functions.T

        v_basis_functions = self.basis_functions[:, 1, :, :, 0].reshape(-1, n * n)
        h_basis_functions = self.basis_functions[:, 2, :, :, 0].reshape(-1, n * n)

        ym_mm = v_basis_functions @ np.diag(self.weights_2D.ravel()) @ h_basis_functions.T
        ym_mm += h_basis_functions @ np.diag(self.weights_2D.ravel()) @ v_basis_functions.T
        ym_mm += v_basis_functions @ np.diag(self.weights_2D.ravel()) @ v_basis_functions.T
        ym_mm += h_basis_functions @ np.diag(self.weights_2D.ravel()) @ h_basis_functions.T

        # c
        self.M1 = ic_mm
        self.Mx = self.M1 + 0.5 * self.x_cfl * (xm_mm - xp_mm)
        self.My = self.M1 + 0.5 * self.y_cfl * (ym_mm - yp_mm)
        self.M = self.Mx + 0.5 * self.y_cfl * (ym_mm - yp_mm)

        self.M1_lu = lu_factor(self.M1)

        self.Mx_lu = lu_factor(self.Mx)
        self.My_lu = lu_factor(self.My)
        self.M_lu = lu_factor(self.M)

    def get_vars(self, arr):
        return (arr[:, :, i] for i in range(3))

    def norm(self, u, v, h):
        return np.sqrt(self.integrate(u ** 2 + v ** 2 + h ** 2))

    def time_step(self, verbose=False, use_lu=True):

        def _first_predictor_bdry():
            u_bdry_integrals[:, :, 0] += self.u / self.weights_x[-1]
            v_bdry_integrals[:, :, 0] += self.v / self.weights_x[-1]
            h_bdry_integrals[:, :, 0] += self.h / self.weights_x[-1]

        def _xbdry(xp, xm, arr):
            u_pred, v_pred, h_pred = self.get_vars(arr)

            # x boundaries
            u_bdry_integrals[xp] += 0.5 * self.x_cfl * h_pred[xm] / self.weights_x[-1]
            u_bdry_integrals[xp] += 0.5 * self.x_cfl * u_pred[xm] / self.weights_x[-1]

            u_bdry_integrals[xm] -= 0.5 * self.x_cfl * h_pred[xp] / self.weights_x[-1]
            u_bdry_integrals[xm] += 0.5 * self.x_cfl * u_pred[xp] / self.weights_x[-1]

            h_bdry_integrals[xp] += 0.5 * self.x_cfl * u_pred[xm] / self.weights_x[-1]
            h_bdry_integrals[xp] += 0.5 * self.x_cfl * h_pred[xm] / self.weights_x[-1]

            h_bdry_integrals[xm] -= 0.5 * self.x_cfl * u_pred[xp] / self.weights_x[-1]
            h_bdry_integrals[xm] += 0.5 * self.x_cfl * h_pred[xp] / self.weights_x[-1]

        def _ybdry(yp, ym, arr):
            u_pred, v_pred, h_pred = self.get_vars(arr)

            # y boundaries
            v_bdry_integrals[yp] += 0.5 * self.y_cfl * h_pred[ym] / self.weights_x[-1]
            v_bdry_integrals[yp] += 0.5 * self.y_cfl * v_pred[ym] / self.weights_x[-1]

            v_bdry_integrals[ym] -= 0.5 * self.y_cfl * h_pred[yp] / self.weights_x[-1]
            v_bdry_integrals[ym] += 0.5 * self.y_cfl * v_pred[yp] / self.weights_x[-1]

            h_bdry_integrals[yp] += 0.5 * self.y_cfl * v_pred[ym] / self.weights_x[-1]
            h_bdry_integrals[yp] += 0.5 * self.y_cfl * h_pred[ym] / self.weights_x[-1]

            h_bdry_integrals[ym] -= 0.5 * self.y_cfl * v_pred[yp] / self.weights_x[-1]
            h_bdry_integrals[ym] += 0.5 * self.y_cfl * h_pred[yp] / self.weights_x[-1]

        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, 3, n, n, n))

        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)

        ###### first predictor
        bdry_integrals[:] = 0.0
        _first_predictor_bdry()

        rhs = np.einsum('abcdef,gcdef->abg', bdry_integrals, self.basis_functions_weighted)
        rhs = rhs.reshape(self.nx * self.ny, -1).transpose()
        state_pred = lu_solve(self.M1_lu, rhs).transpose().copy()
        state_pred = np.einsum('abc,cdefg->abdefg', state_pred.reshape(self.nx, self.ny, -1), self.basis_functions)

        ###### corrector
        for _ in range(3):
            bdry_integrals[:] = 0.0
            _first_predictor_bdry()

            _xbdry(self.xp_int, self.xm_int, state_pred)
            _xbdry(self.xp_ext, self.xm_ext, state_pred)

            _ybdry(self.yp_int, self.ym_int, state_pred)
            _ybdry(self.yp_ext, self.ym_ext, state_pred)

            rhs = np.einsum('abcdef,gcdef->abg', bdry_integrals, self.basis_functions_weighted)
            rhs = rhs.reshape(self.nx * self.ny, -1).transpose()
            state_pred = lu_solve(self.M_lu, rhs).transpose().copy()
            state_pred = np.einsum('abc,cdefg->abdefg', state_pred.reshape(self.nx, self.ny, -1), self.basis_functions)

        self.state[:] = state_pred[:, :, :, -1]
        self.time += self.dt

    # def time_step(self, verbose=False, use_lu=True):
    #
    #     def _first_predictor_bdry():
    #         u_bdry_integrals[:, :, 0] += self.u / self.weights_x[-1]
    #         v_bdry_integrals[:, :, 0] += self.v / self.weights_x[-1]
    #         h_bdry_integrals[:, :, 0] += self.h / self.weights_x[-1]
    #
    #
    #     n = self.poly_order + 1
    #     bdry_integrals = np.zeros((self.nx, self.ny, 3, n, n, n))
    #
    #     u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
    #
    #     ###### first predictor
    #     bdry_integrals[:] = 0.0
    #     _first_predictor_bdry()
    #
    #     rhs = np.einsum('abcdef,gcdef->abg', bdry_integrals, self.basis_functions_weighted)
    #     rhs = rhs.reshape(self.nx * self.ny, -1).transpose()
    #     state_pred = lu_solve(self.M1_lu, rhs).transpose().copy()
    #     state_pred = np.einsum('abc,cdefg->abdefg', state_pred.reshape(self.nx, self.ny, -1), self.basis_functions)
    #
    #     ###### corrector
    #     bdry_integrals = self.corrector(state_pred)
    #     diff = (bdry_integrals * self.weights_x[None, None, None, :, None, None]).sum(axis=3)
    #
    #     # self.state[:] += diff
    #     self.state[:] = state_pred[:, :, :, -1] + diff

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

        # u_bdry_integrals -= self.x_cfl * self.ddxi(h_pred)
        # v_bdry_integrals -= self.y_cfl * self.ddeta(h_pred)
        # h_bdry_integrals -= self.x_cfl * self.ddxi(u_pred) + self.y_cfl * self.ddeta(v_pred)

        return bdry_integrals