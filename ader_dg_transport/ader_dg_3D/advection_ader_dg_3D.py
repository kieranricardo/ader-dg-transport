import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_3D.base_ader_dg_3D import BaseADERDG3D


class AdvectionAderDG3D(BaseADERDG3D):

    def __init__(self, xlim, ylim, zlim, nx, ny, nz, poly_order, cx, cy, cz, dt):

        BaseADERDG3D.__init__(self, xlim, ylim, zlim, nx, ny, nz, poly_order)

        self.cx = cx
        self.cy = cy
        self.cz = cz

        self.dt = dt

        self.x_cfl = self.cx * self.dt / self.dx
        self.y_cfl = self.cy * self.dt / self.dy
        self.z_cfl = self.cz * self.dt / self.dz

        self.u = np.zeros_like(self.xs[:, :, :, 0, :])
        self.u_prev = np.zeros_like(self.xs)
        self.time = 0.0

        (
            Dt, Dx, Dy, Dz, first_space_integral, last_space_integral,
            xm_integral, xp_integral, ym_integral, yp_integral, zm_integral, zp_integral
        ) = self.get_matrices()

        self.M1 = (Dt + self.x_cfl * Dx + self.y_cfl * Dy + self.z_cfl * Dz) + first_space_integral
        self.M1_lu = lu_factor(self.M1)

        # first stage matrices
        self.Mx = np.copy(self.M1)
        self.Mx += self.x_cfl * (xm_integral - xp_integral)
        self.Mx_lu = lu_factor(self.Mx)

        self.My = np.copy(self.M1)
        self.My += self.y_cfl * (ym_integral - yp_integral)
        self.My_lu = lu_factor(self.My)

        self.Mz = np.copy(self.M1)
        self.Mz += self.z_cfl * (zm_integral - zp_integral)
        self.Mz_lu = lu_factor(self.Mz)

        # second stage matrices
        self.Mxy = np.copy(self.My)
        self.Mxy += self.x_cfl * (xm_integral - xp_integral)
        self.Mxy_lu = lu_factor(self.Mxy)

        self.Mxz = np.copy(self.Mz)
        self.Mxz += self.x_cfl * (xm_integral - xp_integral)
        self.Mxz_lu = lu_factor(self.Mxz)

        self.Myz = np.copy(self.Mz)
        self.Myz += self.y_cfl * (ym_integral - yp_integral)
        self.Myz_lu = lu_factor(self.Myz)

        # corrector
        self.M = np.copy(self.M1)
        self.M += self.x_cfl * (xm_integral - xp_integral)
        self.M += self.y_cfl * (ym_integral - yp_integral)
        self.M += self.z_cfl * (zm_integral - zp_integral)
        self.M_lu = lu_factor(self.M)


    def norm(self, arr):
        return np.sqrt(self.integrate(arr ** 2))

    def set_initial_condition(self, u_in):
        self.u[:] = u_in

    def time_step(self, rhs_in=None, tol=1e-8, verbose=False, maxiter=20):

        if rhs_in is None:
            rhs_in = self.get_rhs(self.u)

        bdry_integrals = np.zeros_like(self.xs)

        dx, dy, dz, dt = self.dx, self.dy, self.dz, self.dt

        def _bdry(u_tmp, ip, im, c, dl):
            fluxp = (c * dt / dl * u_tmp)[ip]
            fluxm = (c * dt / dl * u_tmp)[im]

            cup = (abs(c) * dt / dl * u_tmp)[ip]
            cum = (abs(c) * dt / dl * u_tmp)[im]

            num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (cup - cum)
            bdry_integrals[ip] += num_flux / self.weights_x[-1]
            bdry_integrals[im] -= num_flux / self.weights_x[-1]

        def _predictor_loop(u_tmp, bdry_integrals, M_lu, u_xbdry=None, u_ybdry=None, u_zbdry=None):

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

            rhs = bdry_integrals.reshape(self.nx * self.ny * self.nz, -1).transpose()
            u_tmp[:] = lu_solve(M_lu, rhs, overwrite_b=True, check_finite=False).transpose().reshape(self.xs.shape)

            return 1


        iterations = []
        #############################
        # first predictor - no fluxes
        u_pred_1 = np.ones_like(self.xs) * self.u[:, :, :, None]
        i1 = _predictor_loop(u_pred_1, bdry_integrals, M_lu=self.M1_lu, u_xbdry=None, u_ybdry=None, u_zbdry=None)
        iterations.append(i1)

        #############################
        # second predictor - use 1D fluxes
        u_pred_x = np.copy(u_pred_1)
        i1 = _predictor_loop(u_pred_x, bdry_integrals, M_lu=self.Mx_lu, u_xbdry=u_pred_1, u_ybdry=None, u_zbdry=None)
        iterations.append(i1)

        u_pred_y = np.copy(u_pred_1)
        i1 = _predictor_loop(u_pred_y, bdry_integrals, M_lu=self.My_lu, u_xbdry=None, u_ybdry=u_pred_1, u_zbdry=None)
        iterations.append(i1)

        u_pred_z = np.copy(u_pred_1)
        i1 = _predictor_loop(u_pred_z, bdry_integrals, M_lu=self.Mz_lu, u_xbdry=None, u_ybdry=None, u_zbdry=u_pred_1)
        iterations.append(i1)

        #############################
        # third predictor - use 2D fluxes
        u_pred_xy = np.ones_like(self.xs) * self.u[:, :, :, None]
        i1 = _predictor_loop(u_pred_xy, bdry_integrals, M_lu=self.Mxy_lu, u_xbdry=u_pred_y, u_ybdry=u_pred_x, u_zbdry=None)
        iterations.append(i1)

        u_pred_xz = np.ones_like(self.xs) * self.u[:, :, :, None]
        i1 = _predictor_loop(u_pred_xz, bdry_integrals, M_lu=self.Mxz_lu, u_xbdry=u_pred_z, u_ybdry=None, u_zbdry=u_pred_x)
        iterations.append(i1)

        u_pred_yz = np.ones_like(self.xs) * self.u[:, :, :, None]
        i1 = _predictor_loop(u_pred_yz, bdry_integrals, M_lu=self.Myz_lu, u_xbdry=None, u_ybdry=u_pred_z, u_zbdry=u_pred_y)
        iterations.append(i1)

        #############################
        # corrector - all fluxes

        u_out = np.copy(u_pred_1)
        i1 = _predictor_loop(u_out, bdry_integrals, M_lu=self.M_lu, u_xbdry=u_pred_yz, u_ybdry=u_pred_xz, u_zbdry=u_pred_xy)
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