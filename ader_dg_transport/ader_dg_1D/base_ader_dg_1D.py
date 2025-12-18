import numpy as np
from ader_dg_transport.utils import gll, lagrange1st


class BaseADERDG1D:

    def __init__(self, xlim, nx, poly_order):
        self.xlim = xlim
        self.nx = nx
        self.poly_order = poly_order

        self.dx = self.xlim / self.nx

        xis_, self.weights_x = gll(poly_order, iterative=True)
        taus_, self.weights_t = gll(poly_order, iterative=True)
        xis = 0 * taus_[:, None] + xis_[None]

        self.D = lagrange1st(self.poly_order, xis_).transpose().copy()

        xs = np.linspace(0, 1, nx + 1)
        dxi = np.diff(xs).mean()

        xis = xs[:-1, None, None] + (xis[None, :] + 1) * 0.5 * dxi
        self.xs = xlim * xis

        self.u = np.zeros_like(self.xs[:, 0, :])
        self.time = 0.0

    def get_matrices(self):
        n = self.poly_order + 1
        to_st_first = np.zeros((n * n, n))
        for i in range(n):
            to_st_first[i, i] = 1.0

        to_st_last = np.zeros((n * n, n))
        for i in range(n):
            to_st_last[-(n - i), i] = 1.0

        from_st_first = np.zeros((n, n * n))
        for i in range(n):
            from_st_first[i, i] = 1.0

        from_st_last = np.zeros((n, n * n))
        for i in range(n):
            from_st_last[i, -(n - i)] = 1.0

        I = np.eye(n)

        assert np.allclose(from_st_first @ to_st_first, I)
        assert np.allclose(from_st_last @ to_st_last, I)

        xs, w_x = gll(self.poly_order, iterative=True)
        D = lagrange1st(self.poly_order, xs).transpose()

        Dx = np.zeros((n, n, n, n))
        Dt = np.zeros((n, n, n, n))

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    val = D[i, j]
                    Dx[k, i, k, j] = val
                    Dt[i, k, j, k] = val

        assert np.allclose(Dt.swapaxes(0, 1).swapaxes(2, 3), Dx)
        Dx = Dx.reshape((D.size, D.size))
        Dt = Dt.reshape(Dx.shape)

        assert np.allclose(from_st_first @ Dx @ to_st_first, D)
        assert np.allclose(from_st_last @ Dx @ to_st_last, D)

        volume_integral = np.zeros((n, n, n, n))

        left_time_integral = np.zeros((n, n, n, n))  # dt when x = -1
        right_time_integral = np.zeros((n, n, n, n))  # dt when x = 1

        first_space_integral = np.zeros((n, n, n, n))  # dx when t = -1
        last_space_integral = np.zeros((n, n, n, n))  # dx when t = 1

        pick_x0_t1 = np.zeros((n, n, n, n))
        pick_x0_t1[:, 0, -1, 0] = 1.0

        right_to_left = np.zeros((n, n, n, n))
        last_to_first = np.zeros((n, n, n, n))

        for i in range(n):
            for j in range(n):
                volume_integral[i, j, i, j] = w_x[i] * w_x[j]

            left_time_integral[i, 0, i, 0] = w_x[i]
            right_time_integral[i, -1, i, -1] = w_x[i]

            first_space_integral[0, i, 0, i] = w_x[i]
            last_space_integral[-1, i, -1, i] = w_x[i]

            right_to_left[i, 0, i, -1] = 1.0
            last_to_first[0, i, -1, i] = 1.0

        volume_integral = volume_integral.reshape(Dx.shape)

        left_time_integral = left_time_integral.reshape(Dx.shape)
        right_time_integral = right_time_integral.reshape(Dx.shape)

        first_space_integral = first_space_integral.reshape(Dx.shape)
        last_space_integral = last_space_integral.reshape(Dx.shape)

        pick_x0_t1 = pick_x0_t1.reshape(Dx.shape)

        right_to_left = right_to_left.reshape(Dx.shape)
        last_to_first = last_to_first.reshape(Dx.shape)

        assert np.allclose(to_st_first @ from_st_last, last_to_first)

        assert np.allclose(left_time_integral @ right_to_left, right_to_left @ right_time_integral)
        assert np.allclose(first_space_integral @ last_to_first, last_to_first @ last_space_integral)

        return (to_st_first, to_st_last, from_st_first, from_st_last, Dx, Dt,
                volume_integral, left_time_integral, right_time_integral, first_space_integral, last_space_integral,
                pick_x0_t1, right_to_left, last_to_first
                )

    def integrate(self, arr):
        return (arr * self.weights_x[None, :]).sum() * 0.5 * self.dx

    def set_initial_condition(self, u_in):
        self.u[:] = u_in

    def ddxi(self, arr):
        return np.einsum('ab,ecb->eca', self.D, arr)