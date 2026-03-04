import numpy as np
from ader_dg_transport.utils import gll, lagrange1st
from scipy.interpolate import lagrange


class BaseDG2D:

    def __init__(self, xlim, ylim, nx, ny, poly_order, multigrid=False):
        self.xlim = xlim
        self.ylim = ylim

        self.nx = nx
        self.ny = ny
        self.poly_order = poly_order

        self.dx = self.xlim / self.nx
        self.dy = self.ylim / self.ny

        xis_, self.weights_x = gll(poly_order, iterative=True)
        etas_, self.weights_x = gll(poly_order, iterative=True)
        self.D = lagrange1st(self.poly_order, xis_).transpose().copy()

        self.quad_points = np.copy(xis_)

        self.weights_2D = self.weights_x[:, None] * self.weights_x[None, :]

        xis = xis_[:, None] + 0 * etas_[None, :]
        etas = 0 * xis_[:, None] + etas_[None, :]

        xs_ = np.linspace(0, 1, self.nx + 1)
        ys_ = np.linspace(0, 1, self.ny + 1)

        xs = 0 * ys_[None, :] + xs_[:, None]
        ys = ys_[None, :] + 0 * xs_[:, None]

        dxi = np.diff(xs_).mean()
        deta = np.diff(ys_).mean()

        self.xis = xs[:-1, :-1, None, None] + (xis[None, None, :] + 1) * 0.5 * dxi
        self.etas = ys[:-1, :-1, None, None] + (etas[None, None, :] + 1) * 0.5 * deta

        self.xs = self.xlim * self.xis
        self.ys = self.ylim * self.etas

        # interface boundary indices
        self.xp_int = (slice(1, None), slice(None), 0, slice(None))
        self.xm_int = (slice(0, -1), slice(None), -1, slice(None))
        self.yp_int = (slice(None), slice(1, None), slice(None), 0)
        self.ym_int = (slice(None), slice(0, -1), slice(None), -1)
        # domain boundary interfaces
        self.xp_ext = (0, slice(None), 0, slice(None))
        self.xm_ext = (-1, slice(None), -1, slice(None))
        self.yp_ext = (slice(None), 0, slice(None), 0)
        self.ym_ext = (slice(None), -1, slice(None), -1)

    def integrate(self, arr):
        return (arr * self.weights_2D[None, None]).sum() * 0.25 * self.dx * self.dy

    def cell_integrate(self, arr):
        return (arr * self.weights_2D[None, None]).sum(axis=(2, 3)) * 0.25 * self.dx * self.dy

    def ddxi(self, arr):
        return np.einsum('ab,ecbd->ecad', self.D, arr)

    def ddeta(self, arr):
        return np.einsum('ab,ecdb->ecda', self.D, arr)

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

    def apply_x_matrix(self, mat, arr):
        return np.einsum('ab,ecbd->ecad', mat, arr)

    def apply_y_matrix(self, mat, arr):
        return np.einsum('ab,ecdb->ecda', mat, arr)

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


def to_multigrid_points_2D(order):
    x_in, weights = gll(order, iterative=True)
    y_in, _ = gll(order, iterative=True)

    weights_2D = weights[None, :] * weights[:, None]

    x_out_1 = (x_in - 1) * 0.5
    x_out_2 = (x_in + 1) * 0.5
    y_out_1 = (y_in - 1) * 0.5
    y_out_2 = (y_in + 1) * 0.5

    interp_mats = [np.zeros((order + 1, order + 1, order + 1, order + 1)) for _ in range(4)]

    # interp_mat[i, j, k, l] = kl basis at ij point

    for i in range(len(y_out_1)):
        for j in range(len(x_out_1)):
            for k in range(len(y_in)):
                for l in range(len(x_in)):
                    y_data = np.zeros_like(y_in)
                    y_data[k] = 1.0
                    y_poly = lagrange(y_in, y_data)

                    x_data = np.zeros_like(x_in)
                    x_data[l] = 1.0
                    x_poly = lagrange(x_in, x_data)

                    interp_mats[0][i, j, k, l] = y_poly(y_out_1[i]) * x_poly(x_out_1[j])
                    interp_mats[1][i, j, k, l] = y_poly(y_out_1[i]) * x_poly(x_out_2[j])

                    interp_mats[2][i, j, k, l] = y_poly(y_out_2[i]) * x_poly(x_out_1[j])
                    interp_mats[3][i, j, k, l] = y_poly(y_out_2[i]) * x_poly(x_out_2[j])

    n = order + 1
    square_shape = (n ** 2, n ** 2)
    native_shape = (n,) * 4
    coarsen_mats = [mat.reshape(square_shape).transpose().reshape(native_shape) for mat in interp_mats]

    mm = np.zeros(((n ** 2, n ** 2)))
    for i in range(4):
        mat = coarsen_mats[i].reshape((n ** 2, n ** 2))
        mm += 0.25 * mat @ np.diag(weights_2D.ravel()) @ mat.transpose()

    coarsen_inv_mm = np.linalg.inv(mm).reshape((n, n, n, n))
    return interp_mats, coarsen_mats, coarsen_inv_mm, weights_2D


def refine(arr_in, interp_mats):
    order = arr_in.shape[2] - 1
    n = order + 1
    cellsx, cellsz = arr_in.shape[:2]
    out_shape = (2 * cellsx, 2 * cellsz, n, n)
    arr_out = np.zeros(out_shape)

    arr_out[::2, ::2] = np.einsum('abcd,kjcd->kjab', interp_mats[0], arr_in)
    arr_out[::2, 1::2] = np.einsum('abcd,kjcd->kjab', interp_mats[1], arr_in)
    arr_out[1::2, ::2] = np.einsum('abcd,kjcd->kjab', interp_mats[2], arr_in)
    arr_out[1::2, 1::2] = np.einsum('abcd,kjcd->kjab', interp_mats[3], arr_in)

    return arr_out


def coarsen(arr_fine, coarsen_mats, coarsen_inv_mm, coarsen_weights):
    if len(arr_fine.shape) == 5:
        ein_string = 'abcd,zefcd->zefab'
        weights = coarsen_weights[None, None, None]
    elif len(arr_fine.shape) == 4:
        ein_string = 'abcd,efcd->efab'
        weights = coarsen_weights[None, None]
    else:
        raise ValueError(f"arr_coarse: shape mismatch with shape {arr_fine.shape}.")

    arr_w = 0.25 * arr_fine * weights  # multiply by weights and quarter (rescale jacobians)

    arr_coarse = np.einsum(ein_string, coarsen_mats[0], arr_w[::2, ::2])
    arr_coarse += np.einsum(ein_string, coarsen_mats[1], arr_w[::2, 1::2])
    arr_coarse += np.einsum(ein_string, coarsen_mats[2], arr_w[1::2, ::2])
    arr_coarse += np.einsum(ein_string, coarsen_mats[3], arr_w[1::2, 1::2])

    arr_coarse = np.einsum(ein_string, coarsen_inv_mm, arr_coarse)

    return arr_coarse