import numpy as np
from ader_dg_transport.dg_2D.wave_dg_2D import WaveDG2D
from ader_dg_transport.dg_2D.wave_dg_adjoint_2D import WaveAdjointDG2D
from scipy.sparse.linalg import LinearOperator, lsqr


class WaveDGOptimizerReceiver2D:

    def __init__(self, forward_solver, adjoint_solver, ic_state, target_data, nsteps, forcing_func=None, tw=None):

        self.forward_solver = forward_solver
        self.adjoint_solver = adjoint_solver

        if tw is None:
            self.tw = np.ones(nsteps)
        else:
            self.tw = np.copy(tw)

        self.nsteps = nsteps
        self.ic_state = np.copy(ic_state)
        self.target_data = np.copy(target_data)
        self.forcing_func = forcing_func

        self.history_data = np.zeros((nsteps, 4) + self.forward_solver.state.shape)
        self.adjoint_data = np.zeros((nsteps, 4) + self.forward_solver.state.shape)

        self.W = self.forward_solver.weights_2D[0][None, None, None, :] * self.forward_solver.dx * self.forward_solver.dy * 0.25
        self.M = self.adjoint_solver.weights_2D[None, None] * self.adjoint_solver.dy * self.adjoint_solver.dx * 0.25

        self.W_half = np.sqrt(self.W)
        self.M_half = np.sqrt(self.M)

        self.W_half_inv = 1 / self.W_half
        self.M_half_inv = 1 / self.M_half

        self.G = LinearOperator(
            shape=(target_data.size, self.adjoint_solver.c.size),
            matvec=self.G_matvec,
            rmatvec=self.G_rmatvec,
            dtype=np.float64
        )

    def error(self, c_in, fill_gradient=False):

        self.forward_solver.time = 0.0
        self.forward_solver.state[:] = self.ic_state
        self.forward_solver.c[:] = c_in
        self.adjoint_solver.c[:] = self.forward_solver.c

        data = np.zeros_like(self.target_data)
        for i in range(self.nsteps):

            if self.forcing_func is not None:
                ts = np.array([self.forward_solver.time] * 4)
                ts[1] += 0.5 * self.forward_solver.dt
                ts[2] += self.forward_solver.dt
                ts[3] += 0.5 * self.forward_solver.dt

                forcing = self.forcing_func(self.forward_solver.xs, self.forward_solver.ys, ts)
            else:
                forcing = None

            if fill_gradient:
                self.forward_solver.time_step(history_data=self.history_data[i], forcing=forcing)
            else:
                self.forward_solver.time_step(forcing=forcing)

            data[i, :] = self.forward_solver.state[(slice(None),) + self.forward_solver.ym_ext]

        error = (data - self.target_data)[:, :2]
        error *= self.tw[:, None, None, None]

        return error

    def cost_function(self, c_in):
        c_in = c_in.reshape(self.forward_solver.c.shape)
        error = self.error(c_in, fill_gradient=False)

        return 0.5 * (error**2 * self.W).sum()

    def jac_function(self, c_in):
        c_in = c_in.reshape(self.forward_solver.c.shape)
        error = self.error(c_in, fill_gradient=True)

        dedc = self.G_rmatvec(error * self.W_half).reshape(self.forward_solver.c.shape) * self.M_half

        return dedc.ravel()

    def hessp(self, c_in, dc):
        c_in = c_in.reshape(self.forward_solver.c.shape)
        _ = self.error(c_in, fill_gradient=True)

        dc_w = dc.reshape(self.forward_solver.c.shape) * self.M_half

        out = self.G_rmatvec(self.G_matvec(dc_w))
        out = out.reshape(self.forward_solver.c.shape) * self.M_half_inv

        return out.ravel()

    def cost_function_w(self, c_w_in):
        shape = self.forward_solver.c.shape
        c_in = self.M_half_inv * c_w_in.reshape(shape)
        error = self.error(c_in, fill_gradient=False)

        return 0.5 * (error**2 * self.W).sum()

    def jac_function_w(self, c_w_in):

        shape = self.forward_solver.c.shape
        c_in = self.M_half_inv * c_w_in.reshape(shape)
        error = self.error(c_in, fill_gradient=True)

        return self.G_rmatvec(error * self.W_half)

    def hessp_w(self, c_w_in, dc_w):
        shape = self.forward_solver.c.shape
        c_in = self.M_half_inv * c_w_in.reshape(shape)
        _ = self.error(c_in, fill_gradient=True)

        return self.G_rmatvec(self.G_matvec(dc_w))

    def hess_w(self, c_w_in):
        shape = self.forward_solver.c.shape
        c_in = self.M_half_inv * c_w_in.reshape(shape)
        _ = self.error(c_in, fill_gradient=True)

        return self.G.T @ self.G

    def G_matvec(self, dc_vec):

        data = np.zeros_like(self.target_data[:, :2])
        dc = dc_vec.reshape(self.forward_solver.c.shape) * self.M_half_inv
        self.forward_solver.state[:] = 0.0
        for i in range(self.nsteps):
            self.forward_solver.time_step(forcing=self.history_data[i] * dc[None, None])
            data[i, :] = self.forward_solver.state[(slice(None),) + self.forward_solver.ym_ext][:2] * self.tw[i]

        return (self.W_half * data).ravel()

    def G_rmatvec(self, y_vec):
        data = y_vec.reshape(self.target_data[:, :2].shape) * self.W_half_inv

        self.adjoint_solver.state[:] = 0.0

        for i in range(self.nsteps):
            self.adjoint_solver.state[(slice(None),) + self.adjoint_solver.ym_ext][:2] += data[-(i + 1)] * self.tw[-(i + 1)]
            self.adjoint_solver.time_step(stage_data=self.adjoint_data[i])

        tmp = (self.adjoint_data[::-1, ::-1] * self.history_data).sum(axis=(0, 2)) * 0.5 * self.adjoint_solver.dt
        dedc = (tmp[3] + tmp[2] * (1 / 3) + tmp[1] + tmp[0])

        return (self.M_half * dedc).ravel()

    def optimization_step(self, c_in, maxiter=10):

        error = self.error(c_in, fill_gradient=True)

        b = (-error * self.W_half).ravel()

        result = lsqr(self.G, b, atol=1e-10, btol=1e-10, iter_lim=maxiter)
        dc = result[0].reshape(self.forward_solver.c.shape) * self.M_half_inv

        c_out = c_in + dc

        return c_out
