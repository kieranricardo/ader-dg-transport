import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange
from ader_dg_transport.ader_dg_2D.base_ader_dg_2D import BaseADERDG2D
from ader_dg_transport.ader_dg_2D.wave_ader_implicit_iter import WaveAderDG2DImplicitIter


class WaveAderDG2DImplicitAdjointIter(WaveAderDG2DImplicitIter):

    def __init__(self, xlim, ylim, nx, ny, poly_order, c, dt, f=0.0, a=0.5):

        WaveAderDG2DImplicitIter.__init__(self, xlim, ylim, nx, ny, poly_order, c, dt, f=f, a=a)

    def _inner_solver_cpp(self, state_pred, rhs_in, boundaries, maxiter=10, tol=1e-6, verbose=False):
        from ader_dg_transport import dg_kernel_adjoint

        state_pred_new = np.copy(state_pred)

        if boundaries:
            rhs_in = np.copy(rhs_in) - self.wave_forward(0 * state_pred, state_pred)

        u, v, h = self.get_vars(state_pred)
        u_new, v_new, h_new = np.copy(u), np.copy(v), np.copy(h)

        u_rhs_in, v_rhs_in, h_rhs_in = self.get_vars(rhs_in)

        if boundaries:
            _ = dg_kernel_adjoint(u_new, v_new, h_new, u_rhs_in, v_rhs_in, h_rhs_in, self.c, self.D, self.invK, self.x_cfl, self.y_cfl, self.weights_x[-1], maxiter, tol=tol, bdry_flag=1.0)
        else:
            _ = dg_kernel_adjoint(u_new, v_new, h_new, u_rhs_in, v_rhs_in, h_rhs_in, self.c, self.D, self.invK, self.x_cfl, self.y_cfl, self.weights_x[-1], maxiter, tol=tol, bdry_flag=0.0)

        u, v, h = self.get_vars(state_pred_new)
        u[:] = u_new
        v[:] = v_new
        h[:] = h_new

        return state_pred_new

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
        cp, cm = self.c[ip], self.c[im]
        up, hp = cp * u_pred1[ip], cp * h_pred1[ip]
        um, hm = cm * u_pred2[im], cm * h_pred2[im]

        ca = 0.5 * (cp + cm)

        fluxp = -hp # * cp
        fluxm = -hm #* cm
        num_flux = 0.5 * (fluxp + fluxm) - self.a * (up - um)
        u_bdry_integrals[ip] -= cfl * (num_flux - fluxp) / self.weights_x[-1]

        fluxp = -up #* cp
        fluxm = -um #* cm
        num_flux = 0.5 * (fluxp + fluxm) - self.a * (hp - hm)
        h_bdry_integrals[ip] -= cfl * (num_flux - fluxp) / self.weights_x[-1]

        # xm boundaries
        up, hp = cp * u_pred2[ip], cp * h_pred2[ip]
        um, hm = cm * u_pred1[im], cm * h_pred1[im]

        fluxp = -hp #* cp
        fluxm = -hm #* cm
        num_flux = 0.5 * (fluxp + fluxm) - self.a * (up - um)
        u_bdry_integrals[im] += cfl * (num_flux - fluxm) / self.weights_x[-1]

        fluxp = -up #* cp
        fluxm = -um #* cm
        num_flux = 0.5 * (fluxp + fluxm) - self.a * (hp - hm)
        h_bdry_integrals[im] += cfl * (num_flux - fluxm) / self.weights_x[-1]

    def wave_forward_volume(self, state_pred):
        n = self.poly_order + 1
        bdry_integrals = np.zeros((self.nx, self.ny, 3, n, n, n))
        u_bdry_integrals, v_bdry_integrals, h_bdry_integrals = self.get_vars(bdry_integrals)
        u_pred, v_pred, h_pred = self.get_vars(state_pred)

        u_bdry_integrals += -self.x_cfl * self.ddxi(h_pred * self.c)
        v_bdry_integrals += -self.y_cfl * self.ddeta(h_pred * self.c)
        h_bdry_integrals += -self.x_cfl * self.ddxi(self.c * u_pred) - self.y_cfl * self.ddeta(self.c * v_pred)

        return bdry_integrals