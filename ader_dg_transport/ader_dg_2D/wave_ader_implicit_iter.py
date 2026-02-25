import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D


class WaveAderDG2DImplicitIter(BaseADERDG2D):

    def __init__(self, xlim, ylim, nx, ny, poly_order, c, dt, f=0.0, a=0.5):

        BaseADERDG2D.__init__(self, xlim, ylim, nx, ny, poly_order)

        self.dt = dt

        self.x_cfl = self.dt / self.dx
        self.y_cfl = self.dt / self.dy

        self.ts = self.dt * self.taus

        self.time = 0.0

        self.a = a

        n = self.poly_order + 1
        self.state = np.zeros((self.nx, self.ny, 3, n, n))
        self.state_pred = np.zeros((self.nx, self.ny, 3, n, n, n))
        self.c = c * np.ones((self.nx, self.ny, n, n, n))
        self.u = self.state[:, :, 0]
        self.v = self.state[:, :, 1]
        self.h = self.state[:, :, 2]

        self.first = True
        self.state_prev = np.zeros((self.nx, self.ny, 3, n, n, n))

        self.iters = []
        self.diffs = []

    def reset(self):
        self.iters = []
        self.diffs = []
        self.first = True
        self.state[:] = 0.0
        self.state_prev[:] = 0.0
        self.time = 0.0

    def get_vars(self, arr):
        return (arr[:, :, i] for i in range(3))

    def norm(self, u, v, h):
        return np.sqrt(self.integrate(u ** 2 + v ** 2 + h ** 2))


    def preconditioner(self, rhs_in):

        rhs = rhs_in.reshape(self.nx * self.ny, -1).transpose()
        state_out = lu_solve(self.M1_lu, rhs).transpose().copy().reshape(rhs_in.shape)

        return state_out

    def extrapolate(self):
        state_tmp = np.zeros_like(self.state_prev)
        for i in range(len(self.quad_points)):
            for j in range(len(self.quad_points)):
                state_tmp[:, :, :, i] += self.extrapolate_coeffs[j][i] * self.state_prev[:, :, :, j]

        return state_tmp

    def time_step(self, verbose=False, use_lu=True, rhs_in=None, maxiter=10, tol=0, inner_maxiter=1):

        if rhs_in is None:
            rhs_in = self.get_rhs(self.state)

        state_pred = np.ones_like(rhs_in) * self.state[..., None, :, :]

        inner_tol = tol * abs(self.state).max() * 1e-2
        state_pred = self._inner_solver_cpp(state_pred, rhs_in, boundaries=False, verbose=False, maxiter=inner_maxiter, tol=inner_tol)

        for i in range(maxiter):

            state_pred_new = self._inner_solver_cpp(state_pred, rhs_in, boundaries=True, verbose=False, maxiter=inner_maxiter, tol=inner_tol)

            diff = state_pred_new - state_pred
            rel_diff = np.sqrt(self.norm(*self.get_vars(diff))) / np.sqrt(self.norm(*self.get_vars(state_pred_new)))
            state_pred[:] = state_pred_new

            if verbose:
                print("Norm of residual:", np.linalg.norm(self.forward(state_pred) - rhs_in))
                print()

            if rel_diff < tol:
                if verbose:
                    print(f'Exiting after {i} iterations.')
                break

        if rel_diff >= tol:
            print(f'Warning: solver tolerance exceeded. Diff={rel_diff}')

        if verbose:
            print("Norm of residual:", np.linalg.norm(self.forward(state_pred) - rhs_in))

        self.state[:] = state_pred[:, :, :, -1]
        self.state_pred[:] = state_pred
        self.time += self.dt

        return 0

    def _inner_solver(self, state_pred, rhs_in, boundaries, maxiter=10, tol=1e-6, verbose=False):
        state_pred_new = np.copy(state_pred)

        if boundaries:
            rhs_in = np.copy(rhs_in) - self.wave_forward(0 * state_pred, state_pred)

        for i in range(maxiter):
            if boundaries:
                bdry_integrals = rhs_in - self.wave_forward(state_pred_new, 0 * state_pred)
            else:
                bdry_integrals = rhs_in - self.wave_forward_volume(state_pred_new)

            u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)

            if verbose:
                lhs = self.forward_time(state_pred_new)
                error = np.linalg.norm(lhs.ravel() - bdry_integrals.ravel()) / np.linalg.norm(bdry_integrals.ravel())
                print("Inner error:", error)

            u, v, h = self.get_vars(state_pred_new)

            u[:] = self.apply_invK(u_bdry_integrals)
            v[:] = self.apply_invK(v_bdry_integrals)
            h[:] = self.apply_invK(h_bdry_integrals)

        return state_pred_new

    def _inner_solver_cpp(self, state_pred, rhs_in, boundaries, maxiter=10, tol=1e-6, verbose=False):
        from ader_dg_transport import ader_dg_wave_2D_kernel

        state_pred_new = np.copy(state_pred)

        if boundaries:
            rhs_in = np.copy(rhs_in) - self.wave_forward(0 * state_pred, state_pred)

        u, v, h = self.get_vars(state_pred)
        u_new, v_new, h_new = np.copy(u), np.copy(v), np.copy(h)

        u_rhs_in, v_rhs_in, h_rhs_in = self.get_vars(rhs_in)

        if boundaries:
            _ = ader_dg_wave_2D_kernel(u_new, v_new, h_new, u_rhs_in, v_rhs_in, h_rhs_in, self.c, self.D, self.invK, self.x_cfl, self.y_cfl, self.weights_x[-1], maxiter, tol=tol, bdry_flag=1.0)
        else:
            _ = ader_dg_wave_2D_kernel(u_new, v_new, h_new, u_rhs_in, v_rhs_in, h_rhs_in, self.c, self.D, self.invK, self.x_cfl, self.y_cfl, self.weights_x[-1], maxiter, tol=tol, bdry_flag=0.0)

        u, v, h = self.get_vars(state_pred_new)
        u[:] = u_new
        v[:] = v_new
        h[:] = h_new

        return state_pred_new

    def _xbdry_corrector(self, bdry_integrals, arr1, arr2, ip, im):
        self._bdry_corrector(bdry_integrals, arr1, arr2, ip, im, mode='x')

    def _ybdry_corrector(self, bdry_integrals, arr1, arr2, ip, im):
        self._bdry_corrector(bdry_integrals, arr1, arr2, ip, im, mode='y')

    def _bdry_corrector(self, bdry_integrals, arr1, arr2, ip, im, mode):

        # use arr1 for same element
        # use arr2 for neighbouring elements

        if mode == 'x':
            u_bdry_integrals, _, h_bdry_integrals = self.get_vars(bdry_integrals)
            u_pred1, _, h_pred1 = self.get_vars(arr1)
            u_pred2, _, h_pred2 = self.get_vars(arr2)
            cfl = self.x_cfl
        elif mode == 'y':
            _, u_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
            _, u_pred1, h_pred1 = self.get_vars(arr1)
            _, u_pred2, h_pred2 = self.get_vars(arr2)
            cfl = self.y_cfl
        else:
            raise RuntimeError('Wrong mode')

        # xp boundaries
        up, hp = u_pred1[ip], h_pred1[ip]
        um, hm = u_pred2[im], h_pred2[im]
        cp, cm = self.c[ip], self.c[im]
        ca = 0.5 * (cp + cm)

        fluxp = hp # * cp
        fluxm = hm #* cm
        num_flux = 0.5 * (fluxp + fluxm) - self.a * (up - um) #* ca
        u_bdry_integrals[ip] -= cfl * (num_flux - fluxp) * cp / self.weights_x[-1]

        fluxp = up #* cp
        fluxm = um #* cm
        num_flux = 0.5 * (fluxp + fluxm) - self.a * (hp - hm) #* ca
        h_bdry_integrals[ip] -= cfl * (num_flux - fluxp) * cp / self.weights_x[-1]

        # xm boundaries
        up, hp = u_pred2[ip], h_pred2[ip]
        um, hm = u_pred1[im], h_pred1[im]

        fluxp = hp #* cp
        fluxm = hm #* cm
        num_flux = 0.5 * (fluxp + fluxm) - self.a * (up - um) #* ca
        u_bdry_integrals[im] += cfl * (num_flux - fluxm) * cm / self.weights_x[-1]

        fluxp = up #* cp
        fluxm = um #* cm
        num_flux = 0.5 * (fluxp + fluxm) - self.a * (hp - hm) #* ca
        h_bdry_integrals[im] += cfl * (num_flux - fluxm) * cm / self.weights_x[-1]

    def wave_forward(self, state_pred1, state_pred2):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, 3, n, n, n))

        self._xbdry_corrector(bdry_integrals, state_pred1, state_pred2, self.xp_int, self.xm_int)
        self._xbdry_corrector(bdry_integrals, state_pred1, state_pred2, self.xp_ext, self.xm_ext)

        self._ybdry_corrector(bdry_integrals, state_pred1, state_pred2, self.yp_int, self.ym_int)
        self._ybdry_corrector(bdry_integrals, state_pred1, state_pred2, self.yp_ext, self.ym_ext)

        # volume terms
        bdry_integrals += self.wave_forward_volume(state_pred1)

        return bdry_integrals

    def wave_forward_volume(self, state_pred):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, 3, n, n, n))
        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, v_pred, h_pred = self.get_vars(state_pred)

        u_bdry_integrals += self.x_cfl * self.ddxi(h_pred) * self.c
        v_bdry_integrals += self.y_cfl * self.ddeta(h_pred) * self.c
        h_bdry_integrals += self.x_cfl * self.ddxi(u_pred) * self.c + self.y_cfl * self.ddeta(v_pred) * self.c

        return bdry_integrals

    def forward(self, state_pred):

        # wave spatial terms
        bdry_integrals = self.wave_forward(state_pred, state_pred)
        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, v_pred, h_pred = self.get_vars(state_pred)

        # time terms
        u_bdry_integrals += self.ddtau(u_pred)
        v_bdry_integrals += self.ddtau(v_pred)
        h_bdry_integrals += self.ddtau(h_pred)

        u_bdry_integrals[:, :, 0] += u_pred[:, :, 0] / self.weights_x[-1]
        v_bdry_integrals[:, :, 0] += v_pred[:, :, 0] / self.weights_x[-1]
        h_bdry_integrals[:, :, 0] += h_pred[:, :, 0] / self.weights_x[-1]

        return bdry_integrals

    def forward_time(self, state_pred):

        # wave spatial terms
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, 3, n, n, n))
        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, v_pred, h_pred = self.get_vars(state_pred)

        # time terms
        u_bdry_integrals += self.ddtau(u_pred)
        v_bdry_integrals += self.ddtau(v_pred)
        h_bdry_integrals += self.ddtau(h_pred)

        u_bdry_integrals[:, :, 0] += u_pred[:, :, 0] / self.weights_x[-1]
        v_bdry_integrals[:, :, 0] += v_pred[:, :, 0] / self.weights_x[-1]
        h_bdry_integrals[:, :, 0] += h_pred[:, :, 0] / self.weights_x[-1]

        return bdry_integrals

    def get_rhs(self, state):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, 3, n, n, n))

        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u, v, h = self.get_vars(state)

        u_bdry_integrals[:, :, 0] += u / self.weights_x[-1]
        v_bdry_integrals[:, :, 0] += v / self.weights_x[-1]
        h_bdry_integrals[:, :, 0] += h / self.weights_x[-1]

        return bdry_integrals
