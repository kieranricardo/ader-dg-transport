import numpy as np
from scipy.linalg import lu_factor, lu_solve
from ader_dg_transport.ader_dg_1D.base_ader_dg_1D import BaseADERDG1D


class AdvectionADERDG1D(BaseADERDG1D):

    def __init__(self, xlim, nx, poly_order, dt, c):

        BaseADERDG1D.__init__(self, xlim, nx, poly_order)
        self.dt = dt
        self.c = c

        self.cfl = self.dt * c / self.dx


        (to_st_first, to_st_last, from_st_first, from_st_last, Dx, Dt,
         volume_integral, left_time_integral, right_time_integral, first_space_integral, last_space_integral,
         pick_x0_t1, right_to_left, last_to_first
         ) = self.get_matrices()

        self.M1 = volume_integral @ (Dt + self.cfl * Dx) + first_space_integral
        self.M1inv = np.linalg.inv(self.M1)
        self.M1_lu = lu_factor(self.M1)

        self.M = volume_integral @ (Dt + self.cfl * Dx) + first_space_integral
        self.M += self.cfl * (left_time_integral - right_time_integral)
        self.M_lu = lu_factor(self.M)

        self.Minv = np.linalg.inv(self.M)

        self.u = np.zeros_like(self.xs[:, 0, :])
        self.time = 0.0

    def time_step(self, verbose=False):
        udx_t0 = np.zeros_like(self.xs)
        udx_t0[:, 0, :] = self.u * self.weights_x[None, :]

        bdry_integrals = np.zeros_like(self.xs)
        bdry_integrals += udx_t0

        #         u_pred = np.einsum('ab,nb->na', self.M1inv, udx_t0.reshape(self.nx, -1)).reshape(self.xs.shape)

        rhs = bdry_integrals.reshape(self.nx, -1).transpose()
        u_pred = lu_solve(self.M1_lu, rhs)
        u_pred = u_pred.transpose().copy().reshape(self.xs.shape)

        bdry_integrals = np.zeros_like(self.xs)

        bdry_integrals[1:, :, 0] += self.cfl * (u_pred[:-1, :, -1] - u_pred[1:, :, 0]) / self.weights_x[-1]
        bdry_integrals[0, :, 0] += self.cfl * (u_pred[-1, :, -1] - u_pred[0, :, 0])/ self.weights_x[-1]

        # bdry_integrals[:, :, -1] -= self.cfl * u_pred[:, :, -1] / self.weights_x[-1]
        bdry_integrals -= self.cfl * self.ddxi(u_pred)

        diff = (bdry_integrals * self.weights_x[None, :, None]).sum(axis=1)

        self.u[:] += diff
        self.time += self.dt

        self.time += self.dt

    def norm(self):
        return np.sqrt(self.integrate(self.u ** 2))

    def ddxi(self, arr):
        return np.einsum('ab,ecb->eca', self.D, arr)
