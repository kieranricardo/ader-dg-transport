import numpy as np
from ader_dg_transport.utils import gll, lagrange1st
from scipy.interpolate import lagrange


class BaseADERDG2D:

    def __init__(self, xlim, ylim, nx, ny, poly_order, multigrid=False):
        self.xlim = xlim
        self.ylim = ylim

        self.nx = nx
        self.ny = ny
        self.poly_order = poly_order

        self.dx = self.xlim / self.nx
        self.dy = self.ylim / self.ny

        taus_, self.weights_t = gll(poly_order, iterative=True)
        xis_, self.weights_x = gll(poly_order, iterative=True)
        etas_, self.weights_x = gll(poly_order, iterative=True)
        self.D = lagrange1st(self.poly_order, xis_).transpose().copy()
        self.K = np.copy(self.D)
        self.K[0, 0] += 1 / self.weights_x[-1]
        self.invK = np.linalg.inv(self.K)

        self.quad_points = np.copy(taus_)

        self.weights_2D = self.weights_x[:, None] * self.weights_x[None, :]

        taus = taus_[:, None, None] + 0 * xis_[None, :, None] + 0 * etas_[None, None, :]
        xis = 0 * taus_[:, None, None] + xis_[None, :, None] + 0 * etas_[None, None, :]
        etas = 0 * taus_[:, None, None] + 0 * xis_[None, :, None] + etas_[None, None, :]

        xs_ = np.linspace(0, 1, self.nx + 1)
        ys_ = np.linspace(0, 1, self.ny + 1)

        xs = 0 * ys_[None, :] + xs_[:, None]
        ys = ys_[None, :] + 0 * xs_[:, None]

        dxi = np.diff(xs_).mean()
        deta = np.diff(ys_).mean()

        self.xis = xs[:-1, :-1, None, None, None] + (xis[None, None, :] + 1) * 0.5 * dxi
        self.etas = ys[:-1, :-1, None, None, None] + (etas[None, None, :] + 1) * 0.5 * deta
        self.taus = 0 * xs[:-1, :-1, None, None, None] + (taus[None, None, :] + 1) * 0.5

        self.xs = self.xlim * self.xis
        self.ys = self.ylim * self.etas

        # interface boundary indices
        self.xp_int = (slice(1, None), slice(None), slice(None), 0, slice(None))
        self.xm_int = (slice(0, -1), slice(None), slice(None), -1, slice(None))
        self.yp_int = (slice(None), slice(1, None), slice(None), slice(None), 0)
        self.ym_int = (slice(None), slice(0, -1), slice(None), slice(None), -1)
        # domain boundary interfaces
        self.xp_ext = (0, slice(None), slice(None), 0, slice(None))
        self.xm_ext = (-1, slice(None), slice(None), -1, slice(None))
        self.yp_ext = (slice(None), 0, slice(None), slice(None), 0)
        self.ym_ext = (slice(None), -1, slice(None), slice(None), -1)

        # extrapolate in time
        self.extrapolate_coeffs = []
        for i in range(len(self.quad_points)):
            y = np.zeros_like(self.quad_points)
            y[i] = 1.0
            poly = lagrange(self.quad_points, y)

            self.extrapolate_coeffs.append(poly(self.quad_points + 2))

    def broadcast_matrix(self, m):

        n = self.poly_order + 1

        shape = (n,) * 6

        mt = np.zeros(shape)
        mx = np.zeros(shape)
        my = np.zeros(shape)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        val = m[i, j]
                        mt[i, k, l, j, k, l] = val
                        mx[k, i, l, k, j, l] = val
                        my[k, l, i, k, l, j] = val

        assert np.allclose(mt, mx.swapaxes(0, 1).swapaxes(3, 4))
        assert np.allclose(mt, my.swapaxes(0, 2).swapaxes(3, 5))

        mt = mt.reshape((n ** 3, n ** 3))
        mx = mx.reshape(mt.shape)
        my = my.reshape(mt.shape)

        return mt, mx, my

    def get_matrices(self):
        n = self.poly_order + 1

        xs, w_x = gll(self.poly_order, iterative=True)
        D = lagrange1st(self.poly_order, xs).transpose()

        shape = (n,) * 6

        Dt, Dx, Dy = self.broadcast_matrix(D)

        volume_integral = np.zeros(shape)

        first_space_integral = np.zeros(shape)  # dx when t = -1
        last_space_integral = np.zeros(shape)  # dx when t = 1

        xm_integral = np.zeros(shape)  # dt when x = -1
        xp_integral = np.zeros(shape)  # dt when x = 1

        ym_integral = np.zeros(shape)  # dt when x = -1
        yp_integral = np.zeros(shape)  # dt when x = 1

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    volume_integral[i, j, k, i, j, k] = w_x[i] * w_x[j] * w_x[k]

                first_space_integral[0, i, j, 0, i, j] = 1 / w_x[-1]
                last_space_integral[-1, i, j, -1, i, j] = 1 / w_x[-1]

                xm_integral[i, 0, j, i, 0, j] = 1 / w_x[-1]
                xp_integral[i, -1, j, i, -1, j] = 1 / w_x[-1]

                ym_integral[i, j, 0, i, j, 0] = 1 / w_x[-1]
                yp_integral[i, j, -1, i, j, -1] = 1 / w_x[-1]

        volume_integral = volume_integral.reshape(Dt.shape)

        #     assert np.allclose(volume_integral, np.diag((w_x[None, None, :] * w_x[None, :, None] *  w_x[:, None, None]).ravel()))

        #     assert np.allclose(first_space_integral[0, :, :, 0].reshape(n**2, n**2), np.diag((w_x[None, :] * w_x[:, None]).ravel()))

        assert np.allclose(first_space_integral, xm_integral.swapaxes(0, 1).swapaxes(3, 4))
        assert np.allclose(last_space_integral, xp_integral.swapaxes(0, 1).swapaxes(3, 4))

        assert np.allclose(first_space_integral, ym_integral.swapaxes(0, 2).swapaxes(3, 5))
        assert np.allclose(last_space_integral, yp_integral.swapaxes(0, 2).swapaxes(3, 5))

        first_space_integral = first_space_integral.reshape(Dt.shape)
        last_space_integral = last_space_integral.reshape(Dt.shape)

        xm_integral = xm_integral.reshape(Dt.shape)
        xp_integral = xp_integral.reshape(Dt.shape)

        ym_integral = ym_integral.reshape(Dt.shape)
        yp_integral = yp_integral.reshape(Dt.shape)

        out = (
            Dt, Dx, Dy, volume_integral, first_space_integral, last_space_integral,
            xm_integral, xp_integral, ym_integral, yp_integral
        )

        return out

    def integrate(self, arr):
        return (arr * self.weights_2D[None, None]).sum() * 0.25 * self.dx * self.dy

    def cell_integrate(self, arr):
        return (arr * self.weights_2D[None, None]).sum(axis=(2, 3)) * 0.25 * self.dx * self.dy

    def coarsen(self, arr_fine):

        if len(arr_fine.shape) == 5:
            ein_string = 'abcd,zefcd->zefab'
            weights = self.weights_2D[None, None, None]
        elif len(arr_fine.shape) == 4:
            ein_string = 'abcd,efcd->efab'
            weights = self.weights_2D[None, None]
        else:
            raise ValueError(f"arr_coarse: shape mismatch with shape {arr_fine.shape}.")

        arr_w = 0.25 * arr_fine * weights # multiply by weights and quarter (rescale jacobians)

        arr_coarse = np.einsum(ein_string, self.coarsen_mats[0], arr_w[::2, ::2])
        arr_coarse += np.einsum(ein_string, self.coarsen_mats[1], arr_w[::2, 1::2])
        arr_coarse += np.einsum(ein_string, self.coarsen_mats[2], arr_w[1::2, ::2])
        arr_coarse += np.einsum(ein_string, self.coarsen_mats[3], arr_w[1::2, 1::2])

        arr_coarse = np.einsum(ein_string, self.coarsen_inv_mm, arr_coarse)

        return arr_coarse

    def refine(self, arr_coarse):

        nx, ny = arr_coarse.shape[:2]

        if len(arr_coarse.shape) == 5:
            ein_string = 'abcd,zefcd->zefab'
        elif len(arr_coarse.shape) == 4:
            ein_string = 'abcd,efcd->efab'
        else:
            raise ValueError(f"arr_coarse: shape mismatch with shape {arr_coarse.shape}.")

        fine_shape = (2 * nx, 2 * ny) + arr_coarse.shape[2:]
        arr_fine = np.zeros(fine_shape)

        arr_fine[::2, ::2] = np.einsum(ein_string, self.refine_mats[0], arr_coarse)
        arr_fine[::2, 1::2] = np.einsum(ein_string, self.refine_mats[1], arr_coarse)
        arr_fine[1::2, ::2] = np.einsum(ein_string, self.refine_mats[2], arr_coarse)
        arr_fine[1::2, 1::2] = np.einsum(ein_string, self.refine_mats[3], arr_coarse)

        return arr_fine

    def ddxi(self, arr):
        return np.einsum('ab,eczbd->eczad', self.D, arr)

    def ddeta(self, arr):
        return np.einsum('ab,eczdb->eczda', self.D, arr)

    def ddtau(self, arr):
        return np.einsum('ab,ecbzd->ecazd', self.D, arr)

    def ddxi0(self, arr):
        return np.einsum('ab,ecbd->ecad', self.D, arr)

    def ddeta0(self, arr):
        return np.einsum('ab,ecdb->ecda', self.D, arr)

    def apply_K(self, arr):
        return np.einsum('ab,ecbzd->ecazd', self.K, arr)

    def apply_invK(self, arr):
        return np.einsum('ab,ecbzd->ecazd', self.invK, arr)

    def ddx(self, arr):
        return self.ddxi(arr) / self.dx

    def ddy(self, arr):
        return self.ddeta(arr) / self.dy

    def ddxi_jumps(self, arr):
        out = self.ddxi(arr)

        ip = self.xp_int
        im = self.xm_int

        num_flux = 0.5 * (arr[ip] + arr[im])
        out[ip] += (num_flux - arr[ip]) / self.weights_x[-1]
        out[im] -= (num_flux - arr[im]) / self.weights_x[-1]

        ip = self.xp_ext
        im = self.xm_ext

        num_flux = 0.5 * (arr[ip] + arr[im])
        out[ip] += (num_flux - arr[ip]) / self.weights_x[-1]
        out[im] -= (num_flux - arr[im]) / self.weights_x[-1]

        return out

    def ddeta_jumps(self, arr):
        out = self.ddeta(arr)

        ip = self.yp_int
        im = self.ym_int

        num_flux = 0.5 * (arr[ip] + arr[im])
        out[ip] += (num_flux - arr[ip]) / self.weights_x[-1]
        out[im] -= (num_flux - arr[im]) / self.weights_x[-1]

        ip = self.yp_ext
        im = self.ym_ext

        num_flux = 0.5 * (arr[ip] + arr[im])
        out[ip] += (num_flux - arr[ip]) / self.weights_x[-1]
        out[im] -= (num_flux - arr[im]) / self.weights_x[-1]

        return out

    def apply_time_matrix(self, mat, arr):
        return np.einsum('ab,ecbzd->ecazd', mat, arr)

    def apply_x_matrix(self, mat, arr):
        return np.einsum('ab,eczbd->eczad', mat, arr)

    def apply_y_matrix(self, mat, arr):
        return np.einsum('ab,eczdb->eczda', mat, arr)

    def project_H1(self, arr):
        arr_out = np.copy(arr)

        ip = self.xp_int
        im = self.xm_int
        arr_out[ip] = 0.5 * (arr_out[ip] + arr_out[im])
        arr_out[im] = arr_out[ip]

        ip = self.yp_int
        im = self.ym_int
        arr_out[ip] = 0.5 * (arr_out[ip] + arr_out[im])
        arr_out[im] = arr_out[ip]

        ip = self.xp_ext
        im = self.xm_ext
        arr_out[ip] = 0.5 * (arr_out[ip] + arr_out[im])
        arr_out[im] = arr_out[ip]

        ip = self.yp_ext
        im = self.ym_ext
        arr_out[ip] = 0.5 * (arr_out[ip] + arr_out[im])
        arr_out[im] = arr_out[ip]

        return arr_out

    def reshape_plot(self, arr):
        return arr.swapaxes(1, 2).reshape(arr.shape[0] * arr.shape[2], -1)

    def plot_solution(self, data, ax, vmin=None, vmax=None, plot_func=None, dim=3):

        x, y = self.xs[:, :, 0].ravel(), self.ys[:, :, 0].ravel()
        return ax.tricontourf(
            x, y, data.ravel(),
            cmap="nipy_spectral", vmin=vmin, vmax=vmax, levels=100
        )


class Interpolate:

    def __init__(self, xrange, yrange, poly_order, nx, ny, n):

        # get xi and eta per cell
        xis_ = np.linspace(-1, 1, n + 1)[:-1]
        etas_ = np.linspace(-1, 1, n + 1)[:-1]

        xis = xis_[:, None] + 0 * etas_[None, :]
        etas = 0 * xis_[:, None] + etas_[None, :]

        # make transform
        [gll_xs, _] = gll(poly_order, iterative=True)

        self.transform = np.zeros((poly_order + 1, poly_order + 1, n, n))

        for i, y_ in enumerate(gll_xs):
            for j, x_ in enumerate(gll_xs):
                y_data = np.zeros_like(gll_xs)
                y_data[i] = 1.0
                y_poly = lagrange(gll_xs, y_data)

                x_data = np.zeros_like(gll_xs)
                x_data[j] = 1.0
                x_poly = lagrange(gll_xs, x_data)

                self.transform[j, i] = x_poly(xis) * y_poly(etas)

        # make x y coords
        xs_ = np.linspace(0, 1, nx + 1)
        ys_ = np.linspace(0, 1, ny + 1)

        xs = 0 * ys_[None, :] + xs_[:, None]
        ys = ys_[None, :] + 0 * xs_[:, None]

        dxi = np.diff(xs_).mean()
        deta = np.diff(ys_).mean()

        xis = xs[:-1, :-1, None, None] + (xis[None, None, :] + 1) * 0.5 * dxi
        etas = ys[:-1, :-1, None, None] + (etas[None, None, :] + 1) * 0.5 * deta

        self.xs = (xrange[1] - xrange[0]) * xis + xrange[0]
        self.ys = (yrange[1] - yrange[0]) * etas + yrange[0]

        self.x_plot = self.reshape_plot(self.xs)
        self.y_plot = self.reshape_plot(self.ys)

    def interpolate(self, data):
        return np.einsum('abcd,cdef->abef', data, self.transform)

    def reshape_plot(self, arr):
        return arr.swapaxes(1, 2).reshape(arr.shape[0] * arr.shape[2], -1)

    def plot_solution(self, data, ax, vmin=None, vmax=None, plot_func=None, dim=3):

        data = self.interpolate(data)
        z_plot = self.reshape_plot(data)

        return ax.contourf(self.x_plot, self.y_plot, z_plot, cmap="nipy_spectral", vmin=vmin, vmax=vmax, levels=1_000)