import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D


class BurgersAderDG2D(BaseADERDG2D):

    def __init__(self, xlim, ylim, nx, ny, poly_order, dt):

        BaseADERDG2D.__init__(self, xlim, ylim, nx, ny, poly_order)

        self.dt = dt
        
        self.ts = self.dt * self.taus

        self.u = np.zeros_like(self.xs[:, :, 0, :])
        self.u_prev = np.zeros_like(self.xs)
        self.time = 0.0

    def norm(self, arr):
        return np.sqrt(self.integrate(arr ** 2))

    def set_initial_condition(self, u_in):
        self.u[:] = u_in

    def time_step(self, rhs_in=None, tol=1e-8, verbose=False):

        if rhs_in is None:
            rhs_in = self.get_rhs(self.u)

        bdry_integrals = np.zeros_like(self.xs)
        rhs = np.zeros_like(bdry_integrals)
        lhs = np.zeros_like(bdry_integrals)

        dx, dy, dt = self.dx, self.dy, self.dt

        def _iter_loop(u, bdry_integrals, xbdry=False, ybdry=False):
            for i in range(100):
                flux = 0.5 * u ** 2
                rhs[:] = bdry_integrals
                #     rhs[:] -= (dt / dx) * self.ddxi(flux) + (dt / dy) * self.ddeta(flux)
                rhs[:] -= (dt / dx) * (1 / 3) * (self.ddxi(u ** 2) + u * self.ddxi(u))
                rhs[:] -= (dt / dy) * (1 / 3) * (self.ddeta(u ** 2) + u * self.ddeta(u))

                if xbdry:
                    rhs[:, :, :, -1] += (dt / dx) * flux[:, :, :, -1] / self.weights_x[-1]
                    rhs[:, :, :, 0] -= (dt / dx) * flux[:, :, :, 0] / self.weights_x[-1]

                if ybdry:
                    rhs[:, :, :, :, -1] += (dt / dy) * flux[:, :, :, :, -1] / self.weights_x[-1]
                    rhs[:, :, :, :, 0] -= (dt / dy) * flux[:, :, :, :, 0] / self.weights_x[-1]

                lhs[:] = self.ddtau(u)
                lhs[:, :, 0] += u[:, :, 0] / self.weights_x[-1]
                error = np.linalg.norm(lhs.ravel() - rhs.ravel()) / np.linalg.norm(rhs.ravel())

                u[:] = self.apply_invK(rhs)

                if error < tol:
                    break

            return i, error

        def _bdrys(u, bdry_integrals, im, ip, dl):
            up = u[ip]
            um = u[im]
            num_flux = (1 / 6) * (um**2 + um * up + up**2) - 0.5 * abs(up) * (up - um)
            # num_flux = 0.5 * (0.5 * um ** 2 + 0.5 * up ** 2) - 0.5 * abs(up) * (up - um)
            bdry_integrals[ip] += (dt / dl) * num_flux / self.weights_x[-1]
            bdry_integrals[im] -= (dt / dl) * num_flux / self.weights_x[-1]

        #############################
        # first predictor - no fluxes
        u_pred_1 = np.ones_like(self.xs) * self.u[:, :, None, :, :]

        bdry_integrals[:] = rhs_in

        i1, error1 = _iter_loop(u_pred_1, bdry_integrals)

        #############################
        # second predictor - x fluxes
        u_pred_2 = np.copy(u_pred_1)

        bdry_integrals[:] = rhs_in
        _bdrys(u_pred_1, bdry_integrals, self.xm_int, self.xp_int, dx)
        _bdrys(u_pred_1, bdry_integrals, self.xm_ext, self.xp_ext, dx)

        i2, error2 = _iter_loop(u_pred_2, bdry_integrals, xbdry=True)

        #############################
        # third predictor - use y fluxes
        u_pred_3 = np.copy(u_pred_1)

        bdry_integrals[:] = rhs_in
        _bdrys(u_pred_1, bdry_integrals, self.ym_int, self.yp_int, dy)
        _bdrys(u_pred_1, bdry_integrals, self.ym_ext, self.yp_ext, dy)

        i3, error3 = _iter_loop(u_pred_3, bdry_integrals, ybdry=True)

        #############################
        # corrector - use x and y fluxes (which y fluxes to use - pred 2 or 3?)

        u_new = np.copy(u_pred_1)

        bdry_integrals[:] = rhs_in
        _bdrys(u_pred_3, bdry_integrals, self.xm_int, self.xp_int, dx)
        _bdrys(u_pred_3, bdry_integrals, self.xm_ext, self.xp_ext, dx)
        _bdrys(u_pred_2, bdry_integrals, self.ym_int, self.yp_int, dy)
        _bdrys(u_pred_2, bdry_integrals, self.ym_ext, self.yp_ext, dy)

        i4, error4 = _iter_loop(u_new, bdry_integrals, xbdry=True, ybdry=True)

        # update array and times
        self.u[:] = u_new[:, :, -1]
        self.u_prev[:] = u_new

        self.time += self.dt

        if verbose:
            print(f'Solver iterations: {i1}, {i2}, {i3}, {i4}. Maxiter is 20 and tol is {tol}.')
            print(f'Solver iterations: {error1}, {error2}, {error3}, {error4}')

        return None

    def extrapolate(self):
        state_tmp = np.zeros_like(self.u_prev)
        for i in range(len(self.quad_points)):
            for j in range(len(self.quad_points)):
                state_tmp[:, :, i] += self.extrapolate_coeffs[j][i] * self.u_prev[:, :, j]

        return state_tmp

    def get_rhs(self, u):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, n, n, n))
        bdry_integrals[:, :, 0] += u / self.weights_x[-1]
        return bdry_integrals

    def reset(self):
        self.u[:] = 0.0
        self.time = 0.0
