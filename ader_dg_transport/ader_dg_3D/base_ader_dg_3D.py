import numpy as np
from ader_dg_transport.utils import gll, lagrange1st
from scipy.interpolate import lagrange


class BaseADERDG3D:

    def __init__(self, xlim, ylim, zlim, nx, ny, nz, poly_order):
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.poly_order = poly_order

        self.dx = self.xlim / self.nx
        self.dy = self.ylim / self.ny
        self.dz = self.zlim / self.nz

        taus_, self.weights_t = gll(poly_order, iterative=True)
        xis_, self.weights_x = gll(poly_order, iterative=True)
        etas_, self.weights_x = gll(poly_order, iterative=True)
        zetas_, self.weights_x = gll(poly_order, iterative=True)

        self.D = lagrange1st(self.poly_order, xis_).transpose()

        self.K = np.copy(self.D)
        self.K[0, 0] += 1 / self.weights_x[-1]
        self.invK = np.linalg.inv(self.K)

        self.quad_points = np.copy(taus_)

        self.weights_3D = self.weights_x[:, None, None] * self.weights_x[None, :, None] * self.weights_x[None, None, :]

        taus  = 1 * taus_[:, None, None, None] + 0 * xis_[None, :, None, None] + 0 * etas_[None, None, :, None] + 0 * zetas_[None, None, None, :]
        xis   = 0 * taus_[:, None, None, None] + 1 * xis_[None, :, None, None] + 0 * etas_[None, None, :, None] + 0 * zetas_[None, None, None, :]
        etas  = 0 * taus_[:, None, None, None] + 0 * xis_[None, :, None, None] + 1 * etas_[None, None, :, None] + 0 * zetas_[None, None, None, :]
        zetas = 0 * taus_[:, None, None, None] + 0 * xis_[None, :, None, None] + 0 * etas_[None, None, :, None] + 1 * zetas_[None, None, None, :]

        xs_ = np.linspace(0, 1, self.nx + 1)
        ys_ = np.linspace(0, 1, self.ny + 1)
        zs_ = np.linspace(0, 1, self.nz + 1)

        xs = 0 * zs_[None, None, :] + 0 * ys_[None, :, None] + 1 * xs_[:, None, None]
        ys = 0 * zs_[None, None, :] + 1 * ys_[None, :, None] + 0 * xs_[:, None, None]
        zs = 1 * zs_[None, None, :] + 0 * ys_[None, :, None] + 0 * xs_[:, None, None]

        dxi = np.diff(xs_).mean()
        deta = np.diff(ys_).mean()
        dzeta = np.diff(zs_).mean()

        self.xis = xs[:-1, :-1, :-1, None, None, None, None] + (xis[None, None, None, :] + 1) * 0.5 * dxi
        self.etas = ys[:-1, :-1, :-1, None, None, None, None] + (etas[None, None, None, :] + 1) * 0.5 * deta
        self.zetas = zs[:-1, :-1, :-1, None, None, None, None] + (zetas[None, None, None, :] + 1) * 0.5 * dzeta
        self.taus = 0 * xs[:-1, :-1, :-1, None, None, None, None] + (taus[None, None, None, :] + 1) * 0.5

        self.xs = self.xlim * self.xis
        self.ys = self.ylim * self.etas
        self.zs = self.zlim * self.zetas

        # interface boundary indices
        self.xp_int = (slice(1, None), slice(None), slice(None), slice(None), 0, slice(None), slice(None))
        self.xm_int = (slice(0, -1), slice(None), slice(None), slice(None), -1, slice(None), slice(None))
        self.yp_int = (slice(None), slice(1, None), slice(None), slice(None), slice(None), 0, slice(None))
        self.ym_int = (slice(None), slice(0, -1), slice(None), slice(None), slice(None), -1, slice(None))
        self.zp_int = (slice(None), slice(None), slice(1, None), slice(None), slice(None), slice(None), 0)
        self.zm_int = (slice(None), slice(None), slice(0, -1), slice(None), slice(None), slice(None), -1)
        # domain boundary interfaces
        self.xp_ext = (0, slice(None), slice(None), slice(None), 0, slice(None), slice(None))
        self.xm_ext = (-1, slice(None), slice(None), slice(None), -1, slice(None), slice(None))
        self.yp_ext = (slice(None), 0, slice(None), slice(None), slice(None), 0, slice(None))
        self.ym_ext = (slice(None), -1, slice(None), slice(None), slice(None), -1, slice(None))
        self.zp_ext = (slice(None), slice(None), 0, slice(None), slice(None), slice(None), 0)
        self.zm_ext = (slice(None), slice(None), -1, slice(None), slice(None), slice(None), -1)

        # extrapolate in time
        self.extrapolate_coeffs = []
        for i in range(len(self.quad_points)):
            y = np.zeros_like(self.quad_points)
            y[i] = 1.0
            poly = lagrange(self.quad_points, y)

            self.extrapolate_coeffs.append(poly(self.quad_points + 2))

    def get_matrices(self):
        n = self.poly_order + 1

        xs, w_x = gll(self.poly_order, iterative=True)
        D = lagrange1st(self.poly_order, xs).transpose()

        shape = (n,) * 8

        Dt = np.zeros(shape)
        Dx = np.zeros(shape)
        Dy = np.zeros(shape)
        Dz = np.zeros(shape)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        for m in range(n):
                            val = D[i, j]
                            Dt[i, k, l, m, j, k, l, m] = val
                            Dx[k, i, l, m, k, j, l, m] = val
                            Dy[k, l, i, m, k, l, j, m] = val
                            Dz[k, l, m, i, k, l, m, j] = val

        assert np.allclose(Dt, Dx.swapaxes(0, 1).swapaxes(4, 5))
        assert np.allclose(Dt, Dy.swapaxes(0, 2).swapaxes(4, 6))
        assert np.allclose(Dt, Dz.swapaxes(0, 3).swapaxes(4, 7))

        Dt = Dt.reshape((n ** 4, n ** 4))
        Dx = Dx.reshape(Dt.shape)
        Dy = Dy.reshape(Dt.shape)
        Dz = Dz.reshape(Dt.shape)

        first_space_integral = np.zeros(shape)  # dvol when t = -1
        last_space_integral = np.zeros(shape)  # dvol when t = 1

        xm_integral = np.zeros(shape)  # dt * dA when x = -1
        xp_integral = np.zeros(shape)  # dt * dA when x = 1

        ym_integral = np.zeros(shape)  # dt * dA when y = -1
        yp_integral = np.zeros(shape)  # dt * dA when y = 1

        zm_integral = np.zeros(shape)  # dt * dA when z = -1
        zp_integral = np.zeros(shape)  # dt * dA when z = 1

        for i in range(n):
            for j in range(n):
                for k in range(n):

                    first_space_integral[0, i, j, k, 0, i, j, k] = 1 / w_x[-1]
                    last_space_integral[-1, i, j, k, -1, i, j, k] = 1 / w_x[-1]

                    xm_integral[i,  0, j, k, i,  0, j, k] = 1 / w_x[-1]
                    xp_integral[i, -1, j, k, i, -1, j, k] = 1 / w_x[-1]

                    ym_integral[i, j,  0, k, i, j,  0, k] = 1 / w_x[-1]
                    yp_integral[i, j, -1, k, i, j, -1, k] = 1 / w_x[-1]

                    zm_integral[i, j, k,  0, i, j, k,  0] = 1 / w_x[-1]
                    zp_integral[i, j, k, -1, i, j, k, -1] = 1 / w_x[-1]


        assert np.allclose(first_space_integral, xm_integral.swapaxes(0, 1).swapaxes(4, 5))
        assert np.allclose(last_space_integral, xp_integral.swapaxes(0, 1).swapaxes(4, 5))

        assert np.allclose(first_space_integral, ym_integral.swapaxes(0, 2).swapaxes(4, 6))
        assert np.allclose(last_space_integral, yp_integral.swapaxes(0, 2).swapaxes(4, 6))

        assert np.allclose(first_space_integral, zm_integral.swapaxes(0, 3).swapaxes(4, 7))
        assert np.allclose(last_space_integral, zp_integral.swapaxes(0, 3).swapaxes(4, 7))

        first_space_integral = first_space_integral.reshape(Dt.shape)
        last_space_integral = last_space_integral.reshape(Dt.shape)

        xm_integral = xm_integral.reshape(Dt.shape)
        xp_integral = xp_integral.reshape(Dt.shape)

        ym_integral = ym_integral.reshape(Dt.shape)
        yp_integral = yp_integral.reshape(Dt.shape)

        zm_integral = zm_integral.reshape(Dt.shape)
        zp_integral = zp_integral.reshape(Dt.shape)

        out = (
            Dt, Dx, Dy, Dz, first_space_integral, last_space_integral,
            xm_integral, xp_integral, ym_integral, yp_integral, zm_integral, zp_integral
        )

        return out

    def integrate(self, arr):
        return (arr * self.weights_3D[None, None, None]).sum() * 0.125 * self.dx * self.dy * self.dz

    def ddxi(self, arr):
        return np.einsum('ab,xyztbcd->xyztacd', self.D, arr)

    def ddeta(self, arr):
        return np.einsum('ab,xyztcbd->xyztcad', self.D, arr)

    def ddzeta(self, arr):
        return np.einsum('ab,xyztcdb->xyztcda', self.D, arr)

    def ddtau(self, arr):
        return np.einsum('at,xyztbcd->xyzabcd', self.D, arr)

    def apply_invK(self, arr):
        return np.einsum('at,xyztbcd->xyzabcd', self.invK, arr)

    def ddx(self, arr):
        return self.ddxi(arr) / self.dx

    def ddy(self, arr):
        return self.ddeta(arr) / self.dy

    def ddz(self, arr):
        return self.ddzeta(arr) / self.dz

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

        ip = self.zp_int
        im = self.zm_int
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

        ip = self.zp_ext
        im = self.zm_ext
        arr_out[ip] = 0.5 * (arr_out[ip] + arr_out[im])
        arr_out[im] = arr_out[ip]

        return arr_out