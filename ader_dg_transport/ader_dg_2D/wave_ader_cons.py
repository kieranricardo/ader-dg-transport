import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D


class WaveAderConsDG2D(BaseADERDG2D):

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
        self.M1 = np.zeros((3 * n ** 3, 3 * n ** 3))

        u_slice = slice(0, sz)
        v_slice = slice(sz, 2 * sz)
        h_slice = slice(2 * sz, 3 * sz)

        self.M1[u_slice, u_slice] += Dt + first_space_integral
        self.M1[v_slice, v_slice] += Dt + first_space_integral
        self.M1[h_slice, h_slice] += Dt + first_space_integral

        # dudt + dhdx = 0
        self.M1[u_slice, h_slice] += self.x_cfl * Dx
        # dvdt + dhdy = 0
        self.M1[v_slice, h_slice] += self.y_cfl * Dy
        # dhdt + dudx + dvdy = 0
        self.M1[h_slice, u_slice] += self.x_cfl * Dx
        self.M1[h_slice, v_slice] += self.y_cfl * Dy

        self.Mx = np.copy(self.M1)
        # dudt + dhdx = 0
        self.Mx[u_slice, h_slice] += self.x_cfl * (xm_integral - xp_integral)
        # dvdt + dhdy = 0
        # dhdt + dudx + dvdy = 0
        self.Mx[h_slice, u_slice] += self.x_cfl * (xm_integral - xp_integral)

        self.My = np.copy(self.M1)
        # dudt + dhdx = 0
        # dvdt + dhdy = 0
        self.My[v_slice, h_slice] += self.y_cfl * (ym_integral - yp_integral)
        # dhdt + dudx + dvdy = 0
        self.My[h_slice, v_slice] += self.y_cfl * (ym_integral - yp_integral)

        self.M = np.copy(self.M1)
        # dudt + dhdx = 0
        self.M[u_slice, h_slice] += self.x_cfl * (xm_integral - xp_integral)
        # dvdt + dhdy = 0
        self.M[v_slice, h_slice] += self.y_cfl * (ym_integral - yp_integral)
        # dhdt + dudx + dvdy = 0
        self.M[h_slice, u_slice] += self.x_cfl * (xm_integral - xp_integral)
        self.M[h_slice, v_slice] += self.y_cfl * (ym_integral - yp_integral)

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
            num_flux = 0.5 * (h_pred[xm] + h_pred[xp]) - 0.5 * (u_pred[xp] - u_pred[xm])
            u_bdry_integrals[xp] += self.x_cfl * num_flux / self.weights_x[-1]
            u_bdry_integrals[xm] -= self.x_cfl * num_flux / self.weights_x[-1]

            num_flux = 0.5 * (u_pred[xm] + u_pred[xp]) - 0.5 * (h_pred[xp] - h_pred[xm])
            h_bdry_integrals[xp] += self.x_cfl * num_flux / self.weights_x[-1]
            h_bdry_integrals[xm] -= self.x_cfl * num_flux / self.weights_x[-1]

        def _ybdry(yp, ym, arr):
            u_pred, v_pred, h_pred = self.get_vars(arr)

            # y boundaries
            num_flux = 0.5 * (h_pred[ym] + h_pred[yp]) - 0.5 * (v_pred[yp] - v_pred[ym])
            v_bdry_integrals[yp] += self.y_cfl * num_flux / self.weights_x[-1]
            v_bdry_integrals[ym] -= self.y_cfl * num_flux / self.weights_x[-1]

            num_flux = 0.5 * (v_pred[ym] + v_pred[yp]) - 0.5 * (h_pred[yp] - h_pred[ym])
            h_bdry_integrals[yp] += self.y_cfl * num_flux / self.weights_x[-1]
            h_bdry_integrals[ym] -= self.y_cfl * num_flux / self.weights_x[-1]


        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, 3, n, n, n))

        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)

        ###### first predictor
        bdry_integrals[:] = 0.0
        _first_predictor_bdry()

        rhs = np.einsum('abcdef,gcdef->abg', bdry_integrals, self.basis_functions_weighted)
        rhs = rhs.reshape(self.nx * self.ny, -1).transpose()
        state_pred_1 = lu_solve(self.M1_lu, rhs).transpose().copy()
        state_pred_1 = np.einsum('abc,cdefg->abefg', state_pred_1.reshape(self.nx, self.ny, -1), self.basis_functions)

        ###### second predictor
        bdry_integrals[:] = 0.0
        _first_predictor_bdry()

        _xbdry(self.xp_int, self.xm_int, state_pred_1)
        _xbdry(self.xp_ext, self.xm_ext, state_pred_1)

        rhs = np.einsum('abcdef,gcdef->abg', bdry_integrals, self.basis_functions_weighted)
        rhs = rhs.reshape(self.nx * self.ny, -1).transpose()
        state_pred_2 = lu_solve(self.Mx_lu, rhs).transpose().copy()
        state_pred_2 = np.einsum('abc,cdefg->abefg', state_pred_2.reshape(self.nx, self.ny, -1), self.basis_functions)

        ###### third predictor
        bdry_integrals[:] = 0.0
        _first_predictor_bdry()

        _ybdry(self.yp_int, self.ym_int, state_pred_1)
        _ybdry(self.yp_ext, self.ym_ext, state_pred_1)

        rhs = np.einsum('abcdef,gcdef->abg', bdry_integrals, self.basis_functions_weighted)
        rhs = rhs.reshape(self.nx * self.ny, -1).transpose()
        state_pred_3 = lu_solve(self.My_lu, rhs).transpose().copy()
        state_pred_3 = np.einsum('abc,cdefg->abefg', state_pred_3.reshape(self.nx, self.ny, -1), self.basis_functions)

        ###### corrector
        bdry_integrals[:] = 0.0
        _first_predictor_bdry()

        _xbdry(self.xp_int, self.xm_int, state_pred_3)
        _xbdry(self.xp_ext, self.xm_ext, state_pred_3)

        _ybdry(self.yp_int, self.ym_int, state_pred_2)
        _ybdry(self.yp_ext, self.ym_ext, state_pred_2)

        rhs = np.einsum('abcdef,gcdef->abg', bdry_integrals, self.basis_functions_weighted)
        rhs = rhs.reshape(self.nx * self.ny, -1).transpose()
        state_pred = lu_solve(self.M_lu, rhs).transpose().copy()
        state_pred = np.einsum('abc,cdefg->abefg', state_pred.reshape(self.nx, self.ny, -1), self.basis_functions)

        self.state[:] = state_pred[:, :, :, -1]

        self.time += self.dt

    # def time_step(self, verbose=False, use_lu=True):
    #
    #     def _first_predictor_bdry():
    #         u_bdry_integrals[:, :, 0] += self.u / self.weights_x[-1]
    #         v_bdry_integrals[:, :, 0] += self.v / self.weights_x[-1]
    #         h_bdry_integrals[:, :, 0] += self.h / self.weights_x[-1]
    #
    #     def _xbdry(xp, xm, arr, phi):
    #         u_pred, v_pred, h_pred = self.get_vars(arr)
    #
    #         # x boundaries
    #         num_flux = 0.5 * (h_pred[xm] + h_pred[xp]) - 0.5 * (u_pred[xp] - u_pred[xm]) - 0.25 * (phi[xp] - phi[xm])
    #         u_bdry_integrals[xp] += self.x_cfl * num_flux / self.weights_x[-1]
    #         u_bdry_integrals[xm] -= self.x_cfl * num_flux / self.weights_x[-1]
    #
    #         num_flux = 0.5 * (u_pred[xm] + u_pred[xp]) - 0.5 * (h_pred[xp] - h_pred[xm]) - 0.25 * (phi[xp] - phi[xm])
    #         h_bdry_integrals[xp] += self.x_cfl * num_flux / self.weights_x[-1]
    #         h_bdry_integrals[xm] -= self.x_cfl * num_flux / self.weights_x[-1]
    #
    #     def _ybdry(yp, ym, arr, phi):
    #         u_pred, v_pred, h_pred = self.get_vars(arr)
    #
    #         # y boundaries
    #         num_flux = 0.5 * (h_pred[ym] + h_pred[yp]) - 0.5 * (v_pred[yp] - v_pred[ym]) - 0.25 * (phi[yp] - phi[ym])
    #         v_bdry_integrals[yp] += self.y_cfl * num_flux / self.weights_x[-1]
    #         v_bdry_integrals[ym] -= self.y_cfl * num_flux / self.weights_x[-1]
    #
    #         num_flux = 0.5 * (v_pred[ym] + v_pred[yp]) - 0.5 * (h_pred[yp] - h_pred[ym]) - 0.25 * (phi[yp] - phi[ym])
    #         h_bdry_integrals[yp] += self.y_cfl * num_flux / self.weights_x[-1]
    #         h_bdry_integrals[ym] -= self.y_cfl * num_flux / self.weights_x[-1]
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
    #     rhs = bdry_integrals.reshape(self.nx * self.ny, -1).transpose()
    #     state_pred = lu_solve(self.M1_lu, rhs).transpose().copy().reshape(bdry_integrals.shape)
    #     _, _, h_pred = self.get_vars(state_pred)
    #
    #     for _ in range(3):
    #
    #         bdry_integrals[:] = 0.0
    #         _first_predictor_bdry()
    #
    #         # phi = self.apply_invK(self.apply_invK(self.ddeta_jumps(self.ddeta_jumps(h_pred_3)))) * self.y_cfl ** 2
    #         phi = self.apply_invK(self.apply_invK(self.ddeta(self.ddeta(h_pred)))) * self.y_cfl ** 2
    #         _xbdry(self.xp_int, self.xm_int, state_pred, phi)
    #         _xbdry(self.xp_ext, self.xm_ext, state_pred, phi)
    #
    #         # phi = self.apply_invK(self.apply_invK(self.ddxi_jumps(self.ddxi_jumps(h_pred_2)))) * self.x_cfl ** 2
    #         phi = self.apply_invK(self.apply_invK(self.ddxi(self.ddxi(h_pred)))) * self.x_cfl ** 2
    #         _ybdry(self.yp_int, self.ym_int, state_pred, phi)
    #         _ybdry(self.yp_ext, self.ym_ext, state_pred, phi)
    #
    #         rhs = bdry_integrals.reshape(self.nx * self.ny, -1).transpose()
    #         state_pred = lu_solve(self.M_lu, rhs).transpose().copy().reshape(bdry_integrals.shape)
    #         _, _, h_pred = self.get_vars(state_pred)
    #
    #     self.state[:] = state_pred[:, :, :, -1]
    #
    #     self.time += self.dt
