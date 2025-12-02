import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport_transport.ader_dg_3D.base_ader_dg_3D import BaseADERDG3D


class AdvectionIterAderDG3D(BaseADERDG3D):

    def __init__(self, xlim, ylim, zlim, nx, ny, nz, poly_order, cx, cy, cz, dt):

        BaseADERDG3D.__init__(self, xlim, ylim, zlim, nx, ny, nz, poly_order)

        self.cx = cx
        self.cy = cy
        self.cz = cz

        self.dt = dt

        self.u = np.zeros_like(self.xs[:, :, :, 0, :])
        self.u_prev = np.zeros_like(self.xs)
        self.time = 0.0

    def norm(self, arr):
        return np.sqrt(self.integrate(arr ** 2))

    def set_initial_condition(self, u_in):
        self.u[:] = u_in

    def time_step(self, rhs_in=None, tol=1e-8, verbose=False, maxiter=20):

        if rhs_in is None:
            rhs_in = self.get_rhs(self.u)

        bdry_integrals = np.zeros_like(self.xs)
        rhs = np.zeros_like(bdry_integrals)
        lhs = np.zeros_like(bdry_integrals)

        dx, dy, dz, dt = self.dx, self.dy, self.dz, self.dt
        cx, cy, cz = self.cx, self.cy, self.cz

        def _bdry(u_tmp, ip, im, c, dl):
            fluxp = (c * dt / dl * u_tmp)[ip]
            fluxm = (c * dt / dl * u_tmp)[im]

            cup = (abs(c) * dt / dl * u_tmp)[ip]
            cum = (abs(c) * dt / dl * u_tmp)[im]

            num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (cup - cum)
            bdry_integrals[ip] += num_flux / self.weights_x[-1]
            bdry_integrals[im] -= num_flux / self.weights_x[-1]

        def _predictor_loop(u_tmp, bdry_integrals, u_xbdry=None, u_ybdry=None, u_zbdry=None):

            bdry_integrals[:] = rhs_in

            if u_xbdry is not None:
                _bdry(u_xbdry, self.xp_int, self.xm_int, self.cx, self.dx)
                _bdry(u_xbdry, self.xp_ext, self.xm_ext, self.cx, self.dx)

            if u_ybdry is not None:
                _bdry(u_ybdry, self.yp_int, self.ym_int, self.cy, self.dy)
                _bdry(u_ybdry, self.yp_ext, self.ym_ext, self.cy, self.dy)

            if u_zbdry is not None:
                _bdry(u_zbdry, self.zp_int, self.zm_int, self.cz, self.dz)
                _bdry(u_zbdry, self.zp_ext, self.zm_ext, self.cz, self.dz)


            for i in range(maxiter):
                rhs[:] = bdry_integrals
                rhs[:] -= dt * (cx * self.ddx(u_tmp) + cy * self.ddy(u_tmp) + cz * self.ddz(u_tmp))

                if u_xbdry is not None:
                    rhs[..., -1, :, :] += (cx * dt / dx * u_tmp)[..., -1, :, :] / self.weights_x[-1]
                    rhs[..., 0, :, :] -= (cx * dt / dx * u_tmp)[..., 0, :, :] / self.weights_x[-1]

                if u_ybdry is not None:
                    rhs[..., -1, :] += (cy * dt / dy * u_tmp)[..., -1, :] / self.weights_x[-1]
                    rhs[..., 0, :] -= (cy * dt / dy * u_tmp)[..., 0, :] / self.weights_x[-1]

                if u_zbdry is not None:
                    rhs[..., -1] += (cz * dt / dz * u_tmp)[..., -1] / self.weights_x[-1]
                    rhs[..., 0] -= (cz * dt / dz * u_tmp)[..., 0] / self.weights_x[-1]

                lhs[:] = self.ddtau(u_tmp)
                lhs[:, :, :, 0] += u_tmp[:, :, :, 0] / self.weights_x[-1]
                error = np.linalg.norm(lhs.ravel() - rhs.ravel()) / np.linalg.norm(rhs.ravel())

                u_tmp[:] = self.apply_invK(rhs)

                if error < tol:
                    break

            # if verbose:
            #     print('Error:', error)

            return i + 1


        iterations = []
        #############################
        # first predictor - no fluxes
        u_pred_1 = np.ones_like(self.xs) * self.u[:, :, :, None]
        i1 = _predictor_loop(u_pred_1, bdry_integrals, u_xbdry=None, u_ybdry=None, u_zbdry=None)
        iterations.append(i1)

        #############################
        # second predictor - use 1D fluxes
        u_pred_x = np.copy(u_pred_1)
        i1 = _predictor_loop(u_pred_x, bdry_integrals, u_xbdry=u_pred_1, u_ybdry=None, u_zbdry=None)
        iterations.append(i1)

        u_pred_y = np.copy(u_pred_1)
        i1 = _predictor_loop(u_pred_y, bdry_integrals, u_xbdry=None, u_ybdry=u_pred_1, u_zbdry=None)
        iterations.append(i1)

        u_pred_z = np.copy(u_pred_1)
        i1 = _predictor_loop(u_pred_z, bdry_integrals, u_xbdry=None, u_ybdry=None, u_zbdry=u_pred_1)
        iterations.append(i1)

        #############################
        # third predictor - use 2D fluxes
        u_pred_xy = np.ones_like(self.xs) * self.u[:, :, :, None]
        i1 = _predictor_loop(u_pred_xy, bdry_integrals, u_xbdry=u_pred_y, u_ybdry=u_pred_x, u_zbdry=None)
        iterations.append(i1)

        u_pred_xz = np.ones_like(self.xs) * self.u[:, :, :, None]
        i1 = _predictor_loop(u_pred_xz, bdry_integrals, u_xbdry=u_pred_z, u_ybdry=None, u_zbdry=u_pred_x)
        iterations.append(i1)

        u_pred_yz = np.ones_like(self.xs) * self.u[:, :, :, None]
        i1 = _predictor_loop(u_pred_yz, bdry_integrals, u_xbdry=None, u_ybdry=u_pred_z, u_zbdry=u_pred_y)
        iterations.append(i1)

        #############################
        # corrector - all fluxes

        u_out = np.copy(u_pred_1)
        i1 = _predictor_loop(u_out, bdry_integrals, u_xbdry=u_pred_yz, u_ybdry=u_pred_xz, u_zbdry=u_pred_xy)
        iterations.append(i1)

        # update array and times
        self.u[:] = u_out[:, :, :, -1]
        self.u_prev[:] = u_out

        self.time += self.dt

        if verbose:
            print(f'Solver iterations: {iterations}. Total iters: {sum(iterations)}. Maxiter is {maxiter} and tol is {tol}.')

        return None

    def extrapolate(self):
        state_tmp = np.zeros_like(self.u_prev)
        for i in range(len(self.quad_points)):
            for j in range(len(self.quad_points)):
                state_tmp[:, :, i] += self.extrapolate_coeffs[j][i] * self.u_prev[:, :, j]

        return state_tmp

    def get_rhs(self, u):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, self.nz, n, n, n, n))
        bdry_integrals[:, :, :, 0] += u / self.weights_x[-1]
        return bdry_integrals

    def reset(self):
        self.u[:] = 0.0
        self.time = 0.0




def _bdry(u_tmp, ip, im, c, dl):
    fluxp = (c * dt / dl * u_tmp)[ip]
    fluxm = (c * dt / dl * u_tmp)[im]

    cup = (abs(c) * dt / dl * u_tmp)[ip]
    cum = (abs(c) * dt / dl * u_tmp)[im]

    num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (cup - cum)
    bdry_integrals[ip] += num_flux / self.weights_x[-1]
    bdry_integrals[im] -= num_flux / self.weights_x[-1]