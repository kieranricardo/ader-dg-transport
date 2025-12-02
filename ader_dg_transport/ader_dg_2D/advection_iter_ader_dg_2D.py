import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D


class AdvectionIterAderDG2D(BaseADERDG2D):

    def __init__(self, xlim, ylim, nx, ny, poly_order, cx, cy, dt=None, cfl=None):

        BaseADERDG2D.__init__(self, xlim, ylim, nx, ny, poly_order)

        self.cx = cx * np.ones_like(self.xs)
        self.cy = cy * np.ones_like(self.xs)

        self.c_extra = 0.0

        if dt is not None:
            self.dt = dt
        elif cfl is not None:
            self.dt = cfl / ((self.cx / self.dx) + (self.cy / self.dy))
        else:
            raise ValueError('cfl and dt')

        self.x_cfl = self.cx * self.dt / self.dx
        self.y_cfl = self.cy * self.dt / self.dy

        self.ts = self.dt * self.taus

        self.u = np.zeros_like(self.xs[:, :, 0, :])
        self.u_prev = np.zeros_like(self.xs)
        self.time = 0.0

        self.tm = (slice(None), slice(None), 0, slice(None), slice(None))

    def norm(self, arr):
        return np.sqrt(self.integrate(arr ** 2))

    def set_initial_condition(self, u_in):
        self.u[:] = u_in

    def time_step(self, rhs_in=None, tol=1e-8, verbose=False, maxiter=20):

        if rhs_in is None:
            rhs_in = self.get_rhs(self.u)

        #############################
        # first predictor - no fluxes
        u_pred_1 = np.ones_like(rhs_in) * self.u[..., None, :, :]
        i1 = self._predictor_loop(u_pred_1, rhs_in, maxiter, tol, u_xbdry=None, u_ybdry=None)

        #############################
        # second predictor - x fluxes
        u_pred_2 = np.copy(u_pred_1)
        i2 = self._predictor_loop(u_pred_2, rhs_in, maxiter, tol, u_xbdry=u_pred_1, u_ybdry=None)

        #############################
        # third predictor - use y fluxes
        u_pred_3 = np.copy(u_pred_1)
        i3 = self._predictor_loop(u_pred_3, rhs_in, maxiter, tol, u_xbdry=None, u_ybdry=u_pred_1)

        #############################
        # corrector - use x and y fluxes
        u_new = np.copy(u_pred_1)
        i4 = self._predictor_loop(u_new, rhs_in, maxiter, tol, u_xbdry=u_pred_3, u_ybdry=u_pred_2)

        # update array and times
        self.u[:] = u_new[..., -1, :, :]
        self.u_prev[:] = u_new

        self.time += self.dt

        if any(i > maxiter for i in (i1, i2, i3, i4)):
            verbose = True

        if verbose:
            print(f'Solver iterations: {i1}, {i2}, {i3}, {i4}. Maxiter is {maxiter} and tol is {tol}.')

        return None

    def _bdry_fluxes(self, u_in, rhs, ip, im, c, dl):

        c_avg = 0.5 * (c[ip] + c[im])

        u_upwind = (c_avg >= 0) * u_in[im] + (c_avg < 0) * u_in[ip]
        num_flux = c_avg * u_upwind
        rhs[ip] += num_flux * self.dt / (dl * self.weights_x[-1])
        rhs[im] -= num_flux * self.dt / (dl * self.weights_x[-1])

    def _bdry_penalty(self, u_in, rhs, ip, im, c, dl):

        c_avg = 0.5 * (c[ip] + c[im])

        rhs[ip] -= c_avg * u_in[ip] * self.dt / (dl * self.weights_x[-1])
        rhs[im] += c_avg * u_in[im] * self.dt / (dl * self.weights_x[-1])

    def _volume_terms(self, u_in, rhs):
        rhs[:] -= self.dt * (self.cx * self.ddx(u_in) + self.cy * self.ddy(u_in))

    def get_bdry_integrals(self, rhs_in, u_xbdry=None, u_ybdry=None):
        bdry_integrals = np.copy(rhs_in)

        # calculate boundary fluxes
        if u_xbdry is not None:
            self._bdry_fluxes(u_xbdry, bdry_integrals, self.xp_int, self.xm_int, self.cx, self.dx)
            self._bdry_fluxes(u_xbdry, bdry_integrals, self.xp_ext, self.xm_ext, self.cx, self.dx)

        if u_ybdry is not None:
            self._bdry_fluxes(u_ybdry, bdry_integrals, self.yp_int, self.ym_int, self.cy, self.dy)
            self._bdry_fluxes(u_ybdry, bdry_integrals, self.yp_ext, self.ym_ext, self.cy, self.dy)

        return bdry_integrals

    def forward(self, u_tmp, u_xbdry=None, u_ybdry=None):

        out = np.zeros_like(u_tmp)
        self._volume_terms(u_tmp, out)

        if u_xbdry is not None:
            self._bdry_penalty(u_tmp, out, self.xp_int, self.xm_int, self.cx, self.dx)
            self._bdry_penalty(u_tmp, out, self.xp_ext, self.xm_ext, self.cx, self.dx)

        if u_ybdry is not None:
            self._bdry_penalty(u_tmp, out, self.yp_int, self.ym_int, self.cy, self.dy)
            self._bdry_penalty(u_tmp, out, self.yp_ext, self.ym_ext, self.cy, self.dy)

        out *= -1

        out += self.apply_K(u_tmp)

        return out

    def _predictor_loop(self, u_tmp, rhs_in, maxiter, tol, u_xbdry=None, u_ybdry=None):

        rhs = np.zeros_like(rhs_in)
        lhs = np.zeros_like(rhs_in)

        bdry_integrals = self.get_bdry_integrals(rhs_in, u_xbdry=u_xbdry, u_ybdry=u_ybdry)

        for i in range(maxiter):
            rhs[:] = bdry_integrals

            self._volume_terms(u_tmp, rhs)

            if u_xbdry is not None:
                self._bdry_penalty(u_tmp, rhs, self.xp_int, self.xm_int, self.cx, self.dx)
                self._bdry_penalty(u_tmp, rhs, self.xp_ext, self.xm_ext, self.cx, self.dx)

            if u_ybdry is not None:
                self._bdry_penalty(u_tmp, rhs, self.yp_int, self.ym_int, self.cy, self.dy)
                self._bdry_penalty(u_tmp, rhs, self.yp_ext, self.ym_ext, self.cy, self.dy)

            lhs[:] = self.ddtau(u_tmp)
            lhs[self.tm] += u_tmp[self.tm] / self.weights_x[-1]
            error = np.linalg.norm(lhs.ravel() - rhs.ravel()) / np.linalg.norm(rhs.ravel())

            u_tmp[:] = self.apply_invK(rhs)

            if error < tol:
                break

        return i + 1

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
