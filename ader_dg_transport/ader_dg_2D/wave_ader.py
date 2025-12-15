import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D


class WaveAderDG2D(BaseADERDG2D):

    def __init__(self, xlim, nx, poly_order, c, dt, f=0.0):

        ylim = xlim
        ny = nx

        BaseADERDG2D.__init__(self, xlim, ylim, nx, ny, poly_order)

        self.c = c
        self.f = f

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

        self.M1[u_slice, v_slice] -= self.f * self.dt * np.eye(n ** 3)
        self.M1[v_slice, u_slice] += self.f * self.dt * np.eye(n ** 3)

        # dudt + dhdx = 0
        self.M1[u_slice, h_slice] += self.x_cfl * Dx
        # dvdt + dhdy = 0
        self.M1[v_slice, h_slice] += self.y_cfl * Dy
        # dhdt + dudx + dvdy = 0
        self.M1[h_slice, u_slice] += self.x_cfl * Dx
        self.M1[h_slice, v_slice] += self.y_cfl * Dy

        self.M_nc = np.copy(self.M1)
        # dudt + dhdx = 0
        self.M_nc[u_slice, h_slice] += 0.5 * self.x_cfl * (xm_integral - xp_integral)
        # dvdt + dhdy = 0
        self.M_nc[v_slice, h_slice] += 0.5 * self.y_cfl * (ym_integral - yp_integral)
        # dhdt + dudx + dvdy = 0
        self.M_nc[h_slice, u_slice] += 0.5 * self.x_cfl * (xm_integral - xp_integral)
        self.M_nc[h_slice, v_slice] += 0.5 * self.y_cfl * (ym_integral - yp_integral)

        # dissipation terms
        # 0.5 * c * u * dy * dt
        self.M_nc[u_slice, u_slice] += 0.5 * self.x_cfl * (xm_integral + xp_integral)
        self.M_nc[v_slice, v_slice] += 0.5 * self.y_cfl * (ym_integral + yp_integral)
        self.M_nc[h_slice, h_slice] += 0.5 * self.x_cfl * (xm_integral + xp_integral)
        self.M_nc[h_slice, h_slice] += 0.5 * self.y_cfl * (ym_integral + yp_integral)

        self.M1_lu = lu_factor(self.M1)
        self.M_nc_lu = lu_factor(self.M_nc)

        self.first = True
        self.state_prev = np.zeros((self.nx, self.ny, 3, n, n, n))

        self.iters = []
        self.diffs = []

        self.fluxes_from_char = None
        self.to_interior = None
        self.from_interior = None
        self.to_char_bdry = None
        self.from_char_bdry = None
        self.static_cond_lu = None

        self.schur_complement()

    def static_condensation_mats(self):

        def get_bdry_idxs(mask):

            idxs = np.where(mask.ravel())[0]

            to_bdry, from_bdry = get_bdry_mats(idxs, mask)

            return idxs, to_bdry, from_bdry

        def get_bdry_mats(idxs, mask):
            to_bdry = np.zeros((idxs.size, mask.size))

            for i, idx in enumerate(idxs):
                to_bdry[i, idx] = 1.0

            from_bdry = to_bdry.transpose().copy()

            return to_bdry, from_bdry

        n = self.poly_order + 1

        # matrix for converting to and from characteristics
        char_bdry_mask = np.zeros((3, n, n, n))

        # u
        char_bdry_mask[0, :, 0, :] = 1.0
        char_bdry_mask[0, :, -1, :] = 1.0

        # v
        char_bdry_mask[1, :, :, 0] = 1.0
        char_bdry_mask[1, :, :, -1] = 1.0

        u_idxs1 = np.zeros(2 * n ** 2)

        u_idxs2, u_idxs4 = np.where(np.ones((n, n)))
        u_idxs2 = np.concatenate([u_idxs2, u_idxs2])
        u_idxs3 = np.concatenate([np.zeros(n ** 2), (n - 1) * np.ones(n ** 2)])
        u_idxs4 = np.concatenate([u_idxs4, u_idxs4])

        u_idxs_list = [u_idxs4, u_idxs3, u_idxs2, u_idxs1]
        stride = 1.0
        u_idxs = 0.0
        for idxs_ in u_idxs_list:
            u_idxs += stride * idxs_
            stride *= n

        u_idxs = u_idxs.astype(np.int64)

        v_idxs1 = np.ones(2 * n ** 2)

        v_idxs2, v_idxs3 = np.where(np.ones((n, n)))
        v_idxs2 = np.concatenate([v_idxs2, v_idxs2])
        v_idxs3 = np.concatenate([v_idxs3, v_idxs3])

        v_idxs4 = np.concatenate([np.zeros(n ** 2), (n - 1) * np.ones(n ** 2)])

        v_idxs_list = [v_idxs4, v_idxs3, v_idxs2, v_idxs1]
        stride = 1.0
        v_idxs = 0.0
        for idxs_ in v_idxs_list:
            v_idxs += stride * idxs_
            stride *= n

        v_idxs = v_idxs.astype(np.int64)

        char_idxs = np.concatenate([u_idxs, v_idxs])

        self.to_char_bdry, self.from_char_bdry = get_bdry_mats(char_idxs, char_bdry_mask)

        interior_mask = 1.0 - char_bdry_mask
        interior_idxs, self.to_interior, self.from_interior = get_bdry_idxs(interior_mask)

        # convert characteristics to fluxes
        fluxes_from_char = np.zeros((3, n, n, n, 4, n, n))

        for i in range(n):
            for j in range(n):
                # u at x = -1
                fluxes_from_char[0, i, 0, j, 0, i, j] = 1.0 * (0.5 * self.x_cfl / self.weights_x[-1])
                # h at x = -1
                fluxes_from_char[2, i, 0, j, 0, i, j] = 1.0 * (0.5 * self.x_cfl / self.weights_x[-1])

                # u at x = 1
                fluxes_from_char[0, i, -1, j, 1, i, j] = 1.0 * (0.5 * self.x_cfl / self.weights_x[-1])
                # h at x = 1
                fluxes_from_char[2, i, -1, j, 1, i, j] = -1.0 * (0.5 * self.x_cfl / self.weights_x[-1])

                # v at y = -1
                fluxes_from_char[1, i, j, 0, 2, i, j] = 1.0 * (0.5 * self.y_cfl / self.weights_x[-1])
                # y at y = -1
                fluxes_from_char[2, i, j, 0, 2, i, j] = 1.0 * (0.5 * self.y_cfl / self.weights_x[-1])

                # v at y = 1
                fluxes_from_char[1, i, j, -1, 3, i, j] = 1.0 * (0.5 * self.y_cfl / self.weights_x[-1])
                # y at y = 1
                fluxes_from_char[2, i, j, -1, 3, i, j] = -1.0 * (0.5 * self.y_cfl / self.weights_x[-1])

        self.fluxes_from_char = fluxes_from_char.reshape((3 * n ** 3, 4 * n ** 2))

        # change u and v bdry points outgoing chars
        to_out_chars = np.eye(3 * n ** 3)
        shape = to_out_chars.shape
        to_out_chars = to_out_chars.reshape((3, n, n, n, 3, n, n, n))

        for i in range(n):
            for j in range(n):
                # x = -1
                # char = (u - h)
                row = to_out_chars[0, i, 0, j]
                row[2, i, 0, j] = -1.0

                # u at x = 1
                # char = (u + h)
                row = to_out_chars[0, i, -1, j]
                row[2, i, -1, j] = 1.0

                # v at y = -1
                # char = (v - h)
                row = to_out_chars[1, i, j, 0]
                row[2, i, j, 0] = -1.0

                # v at x = 1
                # char = (u + h)
                row = to_out_chars[1, i, j, -1]
                row[2, i, j, -1] = 1.0

        self.to_out_chars = to_out_chars.reshape(shape)

        # from incoming chars
        from_out_chars = np.eye(3 * n ** 3)
        shape = from_out_chars.shape
        from_out_chars = from_out_chars.reshape((3, n, n, n, 3, n, n, n))

        for i in range(n):
            for j in range(n):
                # u at x = -1
                # u = (u + h) - h
                row = from_out_chars[0, i, 0, j]
                row[2, i, 0, j] = 1.0

                # u at x = 1
                # u = (u - h) + h
                row = from_out_chars[0, i, -1, j]
                row[2, i, -1, j] = -1.0

                # v at y = -1
                # v = (v + h) - h
                row = from_out_chars[1, i, j, 0]
                row[2, i, j, 0] = 1.0

                # v at x = 1
                # v = (u - h) + h
                row = from_out_chars[1, i, j, -1]
                row[2, i, j, -1] = -1.0

        self.from_out_chars = from_out_chars.reshape(shape)

    def schur_complement(self):

        self.static_condensation_mats()

        M = self.M_nc @ self.from_out_chars

        ## blocks
        A = self.to_interior @ M @ self.from_interior
        B = self.to_interior @ M @ self.from_char_bdry
        C = self.to_char_bdry @ M @ self.from_interior
        D = self.to_char_bdry @ M @ self.from_char_bdry

        ## get rhs
        u = self.to_interior @ self.fluxes_from_char
        v = self.to_char_bdry @ self.fluxes_from_char

        rhs = v - C @ np.linalg.inv(A) @ u

        ## final solution
        static_cond_mat = np.linalg.inv(rhs) @ (D - C @ np.linalg.inv(A) @ B)

        self.static_cond_lu = lu_factor(static_cond_mat)

        self.undo_static_1 = self.from_out_chars @ (self.from_char_bdry + self.from_interior @ (-np.linalg.inv(A) @ B))
        self.undo_static_2 = self.from_out_chars @ self.from_interior @ (np.linalg.inv(A) @ self.to_interior @ self.fluxes_from_char)

    def get_out_chars(self, state_in):
        n = self.poly_order + 1

        u, v, h = self.get_vars(state_in)
        chars_out = np.zeros((self.nx, self.ny, 4, n, n))

        chars_out[:, :, 0] = u[..., 0, :] - h[..., 0, :]
        chars_out[:, :, 1] = u[..., -1, :] + h[..., -1, :]
        chars_out[:, :, 2] = v[..., 0] - h[..., 0]
        chars_out[:, :, 3] = v[..., -1] + h[..., -1]

        return chars_out

    def in_chars_to_fluxes(self, chars_in):
        n = self.poly_order + 1
        fluxes = np.zeros((self.nx, self.ny, 3, n, n, n))

        u, v, h = self.get_vars(fluxes)

        u[..., 0, :] += chars_in[:, :, 0] * (0.5 * self.x_cfl / self.weights_x[-1])
        h[..., 0, :] += chars_in[:, :, 0] * (0.5 * self.x_cfl / self.weights_x[-1])

        u[..., -1, :] += chars_in[:, :, 1] * (0.5 * self.x_cfl / self.weights_x[-1])
        h[..., -1, :] += -chars_in[:, :, 1] * (0.5 * self.x_cfl / self.weights_x[-1])

        v[..., 0] += chars_in[:, :, 2] * (0.5 * self.y_cfl / self.weights_x[-1])
        h[..., 0] += chars_in[:, :, 2] * (0.5 * self.y_cfl / self.weights_x[-1])

        v[..., -1] += chars_in[:, :, 3] * (0.5 * self.y_cfl / self.weights_x[-1])
        h[..., -1] += -chars_in[:, :, 3] * (0.5 * self.y_cfl / self.weights_x[-1])

        return fluxes

    def fast_block_jacobi(self, state_in, rhs_in, N=10, tol=0.0):

        state_0 = self.block_jacobi(0 * state_in, rhs_in=rhs_in)
        out_char_0 = self.get_out_chars(state_0)
        chars_out = self.get_out_chars(state_in)

        n = self.poly_order + 1
        shape = (self.nx, self.ny, 4, n, n)

        chars_out = chars_out.reshape(shape)
        chars_in = np.zeros_like(chars_out)
        prev_chars_out = np.zeros_like(chars_out)

        for iter in range(N):
            for (xp, xm) in ((self.xp_int, self.xm_int), (self.xp_ext, self.xm_ext)):
                ip, im = xp[:2] + (0,), xm[:2] + (1,)

                chars_in[im] = chars_out[ip]
                chars_in[ip] = chars_out[im]

            for (yp, ym) in ((self.yp_int, self.ym_int), (self.yp_ext, self.ym_ext)):
                ip, im = yp[:2] + (2,), ym[:2] + (3,)

                chars_in[im] = chars_out[ip]
                chars_in[ip] = chars_out[im]

            rhs = chars_in.reshape((self.nx * self.ny, -1)).transpose()

            prev_chars_out[:] = chars_out

            chars_out = lu_solve(self.static_cond_lu, rhs).transpose()
            chars_out += out_char_0.reshape(chars_out.shape)
            chars_out = chars_out.reshape(shape)

            diff = abs(prev_chars_out - chars_out).mean() / abs(chars_out).mean()

            if diff < tol:
                break

        chars_out -= out_char_0.reshape(chars_out.shape)

        bdry_integrals = self.in_chars_to_fluxes(chars_in)
        rhs = bdry_integrals.reshape(self.nx * self.ny, -1).transpose()
        state_out = lu_solve(self.M_nc_lu, rhs).transpose().reshape(bdry_integrals.shape)

        state_out += state_0

        return state_out, iter

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

    def _xbdry_nc(self, bdry_integrals, arr, xp, xm):

        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, v_pred, h_pred = self.get_vars(arr)

        # x boundaries
        # flux fluc = 0.5 * (h_pred + h) - 0.5 * (u - u_pred) - h
        u_bdry_integrals[xp] += 0.5 * self.x_cfl * h_pred[xm] / self.weights_x[-1]
        u_bdry_integrals[xp] += 0.5 * self.x_cfl * u_pred[xm] / self.weights_x[-1]

        u_bdry_integrals[xm] -= 0.5 * self.x_cfl * h_pred[xp] / self.weights_x[-1]
        u_bdry_integrals[xm] += 0.5 * self.x_cfl * u_pred[xp] / self.weights_x[-1]

        h_bdry_integrals[xp] += 0.5 * self.x_cfl * u_pred[xm] / self.weights_x[-1]
        h_bdry_integrals[xp] += 0.5 * self.x_cfl * h_pred[xm] / self.weights_x[-1]

        h_bdry_integrals[xm] -= 0.5 * self.x_cfl * u_pred[xp] / self.weights_x[-1]
        h_bdry_integrals[xm] += 0.5 * self.x_cfl * h_pred[xp] / self.weights_x[-1]

    def _ybdry_nc(self, bdry_integrals, arr, yp, ym):

        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
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

    def extrapolate(self):
        state_tmp = np.zeros_like(self.state_prev)
        for i in range(len(self.quad_points)):
            for j in range(len(self.quad_points)):
                state_tmp[:, :, :, i] += self.extrapolate_coeffs[j][i] * self.state_prev[:, :, :, j]

        return state_tmp

    def time_step(self, verbose=False, use_lu=True, multi_grid_pred=None, rhs_in=None, wt=1.0, tol=1e-8):

        if rhs_in is None:
            rhs_in = self.get_rhs(self.state)

        state_pred = self.preconditioner(rhs_in)

        for i in range(3):
            state_pred = self.block_jacobi(state_pred, rhs_in)

        self.state[:] = state_pred[:, :, :, -1]
        self.time += self.dt

        return 0

    def _xbdry_corrector(self, bdry_integrals, arr, xp, xm):

        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, v_pred, h_pred = self.get_vars(arr)

        # x boundaries
        fluxp = h_pred[xp]
        fluxm = h_pred[xm]
        num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (u_pred[xp] - u_pred[xm])
        u_bdry_integrals[xp] -= self.x_cfl * (num_flux - fluxp) / self.weights_x[-1]
        u_bdry_integrals[xm] += self.x_cfl * (num_flux - fluxm) / self.weights_x[-1]

        fluxp = u_pred[xp]
        fluxm = u_pred[xm]
        num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (h_pred[xp] - h_pred[xm])
        h_bdry_integrals[xp] -= self.x_cfl * (num_flux - fluxp) / self.weights_x[-1]
        h_bdry_integrals[xm] += self.x_cfl * (num_flux - fluxm) / self.weights_x[-1]

    def _ybdry_corrector(self, bdry_integrals, arr, yp, ym):

        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, v_pred, h_pred = self.get_vars(arr)

        u_pred, v_pred, h_pred = self.get_vars(arr)
        # y boundaries
        fluxp = h_pred[yp]
        fluxm = h_pred[ym]
        num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (v_pred[yp] - v_pred[ym])
        v_bdry_integrals[yp] -= self.y_cfl * (num_flux - fluxp) / self.weights_x[-1]
        v_bdry_integrals[ym] += self.y_cfl * (num_flux - fluxm) / self.weights_x[-1]

        fluxp = v_pred[yp]
        fluxm = v_pred[ym]
        num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (h_pred[yp] - h_pred[ym])
        h_bdry_integrals[yp] -= self.y_cfl * (num_flux - fluxp) / self.weights_x[-1]
        h_bdry_integrals[ym] += self.y_cfl * (num_flux - fluxm) / self.weights_x[-1]

    def wave_forward(self, state_pred):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, 3, n, n, n))

        self._xbdry_corrector(bdry_integrals, state_pred, self.xp_int, self.xm_int)
        self._xbdry_corrector(bdry_integrals, state_pred, self.xp_ext, self.xm_ext)

        self._ybdry_corrector(bdry_integrals, state_pred, self.yp_int, self.ym_int)
        self._ybdry_corrector(bdry_integrals, state_pred, self.yp_ext, self.ym_ext)

        # volume terms
        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, v_pred, h_pred = self.get_vars(state_pred)

        u_bdry_integrals += self.x_cfl * self.ddxi(h_pred)
        v_bdry_integrals += self.y_cfl * self.ddeta(h_pred)
        h_bdry_integrals += self.x_cfl * self.ddxi(u_pred) + self.y_cfl * self.ddeta(v_pred)

        return bdry_integrals

    def forward(self, state_pred):

        # wave spatial terms
        bdry_integrals = self.wave_forward(state_pred)
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
