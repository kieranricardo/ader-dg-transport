import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D


class AdvectionStdAderDG2D(BaseADERDG2D):

    def __init__(self, xlim, ylim, nx, ny, poly_order, cx, cy, dt=None, cfl=None):

        BaseADERDG2D.__init__(self, xlim, ylim, nx, ny, poly_order)

        self.cx = cx
        self.cy = cy

        if dt is not None:
            self.dt = dt
        elif cfl is not None:
            self.dt = cfl / ((self.cx / self.dx) + (self.cy / self.dy))
        else:
            raise ValueError('cfl and dt')

        self.x_cfl = self.cx * self.dt / self.dx
        self.y_cfl = self.cy * self.dt / self.dy

        self.ts = self.dt * self.taus

        (Dt, Dx, Dy, volume_integral, first_space_integral,
         last_space_integral, xm_integral, xp_integral, ym_integral, yp_integral) = self.get_matrices()

        self.M1 = (Dt + self.x_cfl * Dx + self.y_cfl * Dy) + first_space_integral
        self.M1_lu = lu_factor(self.M1)

        self.u = np.zeros_like(self.xs[:, :, 0, :])
        self.u_prev = np.zeros_like(self.xs)
        self.time = 0.0

        for i in range(self.nx):
            for j in range(self.poly_order + 1):
                assert (self.xs[i, :, :, j] == self.xs[i, :, :, j].ravel()[0]).all()

        for i in range(self.ny):
            for j in range(self.poly_order + 1):
                assert (self.ys[:, i, :, :, j] == self.ys[:, i, :, :, j].ravel()[0]).all()

        for j in range(self.poly_order + 1):
            assert (self.ts[:, :, j] == self.ts[:, :, j].ravel()[0]).all()

    def norm(self, arr):
        return np.sqrt(self.integrate(arr ** 2))

    def set_initial_condition(self, u_in):
        self.u[:] = u_in

    def time_step(self, rhs_in=None, initial_pred=None, verbose=False):

        if rhs_in is None:
            rhs_in = self.get_rhs(self.u)

        bdry_integrals = np.zeros_like(self.xs)

        #############################
        # first predictor - no fluxes

        if initial_pred is None:
            bdry_integrals[:] = rhs_in

            rhs = bdry_integrals.reshape(self.nx * self.ny, -1).transpose()
            u_pred_1 = lu_solve(self.M1_lu, rhs).transpose().copy().reshape(self.xs.shape)
        else:
            u_pred_1 = np.copy(initial_pred)

        #############################
        # second predictor - x fluxes
        bdry_integrals[:] = rhs_in

        ### x-boundary integrals
        bdry_integrals[1:, :, :, 0] += self.x_cfl * u_pred_1[:-1, :, :, -1] / self.weights_x[-1]
        bdry_integrals[:, :, :, -1] -= self.x_cfl * u_pred_1[:, :, :, -1] / self.weights_x[-1]
        bdry_integrals[0, :, :, 0] += self.x_cfl * u_pred_1[-1, :, :, -1] / self.weights_x[-1]

        rhs = bdry_integrals.reshape(self.nx * self.ny, -1).transpose()
        u_pred_2 = lu_solve(self.Mx_lu, rhs).transpose().copy().reshape(self.xs.shape)

        #############################
        # third predictor - use y fluxes
        bdry_integrals[:] = rhs_in

        ### y-boundary integrals
        bdry_integrals[:, 1:, :, :, 0] += self.y_cfl * u_pred_1[:, :-1, :, :, -1] / self.weights_x[-1]
        bdry_integrals[:, :, :, :, -1] -= self.y_cfl * u_pred_1[:, :, :, :, -1] / self.weights_x[-1]
        bdry_integrals[:, 0, :, :, 0] += self.y_cfl * u_pred_1[:, -1, :, :, -1] / self.weights_x[-1]

        rhs = bdry_integrals.reshape(self.nx * self.ny, -1).transpose()
        u_pred_3 = lu_solve(self.My_lu, rhs).transpose().copy().reshape(self.xs.shape)

        #############################
        # corrector - use x and y fluxes (which y fluxes to use - pred 2 or 3?)

        bdry_integrals[:] = rhs_in

        ### x-boundary integrals
        bdry_integrals[1:, :, :, 0] += self.x_cfl * u_pred_3[:-1, :, :, -1] / self.weights_x[-1]
        bdry_integrals[:, :, :, -1] -= self.x_cfl * u_pred_3[:, :, :, -1] / self.weights_x[-1]
        bdry_integrals[0, :, :, 0] += self.x_cfl * u_pred_3[-1, :, :, -1] / self.weights_x[-1]

        ### y-boundary integrals
        bdry_integrals[:, 1:, :, :, 0] += self.y_cfl * u_pred_2[:, :-1, :, :, -1] / self.weights_x[-1]
        bdry_integrals[:, :, :, :, -1] -= self.y_cfl * u_pred_2[:, :, :, :, -1] / self.weights_x[-1]
        bdry_integrals[:, 0, :, :, 0] += self.y_cfl * u_pred_2[:, -1, :, :, -1] / self.weights_x[-1]

        rhs = bdry_integrals.reshape(self.nx * self.ny, -1).transpose()
        u_new = lu_solve(self.M_lu, rhs).transpose().copy().reshape(self.xs.shape)
        self.u[:] = u_new[:, :, -1]
        self.u_prev[:] = u_new

        self.time += self.dt

        adv_terms = self.forward(u_new, u_pred_2, u_pred_3)
        return u_new, adv_terms

    def extrapolate(self):
        state_tmp = np.zeros_like(self.u_prev)
        for i in range(len(self.quad_points)):
            for j in range(len(self.quad_points)):
                state_tmp[:, :, i] += self.extrapolate_coeffs[j][i] * self.u_prev[:, :, j]

        return state_tmp

    def forward(self, u_pred, u_pred_2, u_pred_3):

        bdry_integrals = np.zeros_like(u_pred)

        def _xbdry(xp, xm):
            fluxp = u_pred[xp]
            fluxm = u_pred[xm]
            num_flux = u_pred_3[xm]
            bdry_integrals[xp] -= self.x_cfl * (num_flux - fluxp) / self.weights_x[-1]
            bdry_integrals[xm] += self.x_cfl * (num_flux - fluxm) / self.weights_x[-1]

        def _ybdry(yp, ym):
            fluxp = u_pred[yp]
            fluxm = u_pred[ym]
            num_flux = u_pred_2[ym]
            bdry_integrals[yp] -= self.y_cfl * (num_flux - fluxp) / self.weights_x[-1]
            bdry_integrals[ym] += self.y_cfl * (num_flux - fluxm) / self.weights_x[-1]

        _xbdry(self.xp_int, self.xm_int)
        _xbdry(self.xp_ext, self.xm_ext)

        _ybdry(self.yp_int, self.ym_int)
        _ybdry(self.yp_ext, self.ym_ext)

        bdry_integrals += self.x_cfl * self.ddxi(u_pred) + self.y_cfl * self.ddeta(u_pred)

        return bdry_integrals

    def get_rhs(self, u):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, n, n, n))
        bdry_integrals[:, :, 0] += u / self.weights_x[-1]
        return bdry_integrals

    def reset(self):
        self.u[:] = 0.0
        self.time = 0.0
