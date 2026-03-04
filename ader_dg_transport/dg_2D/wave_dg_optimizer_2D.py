import numpy as np
from ader_dg_transport.dg_2D.wave_dg_2D import WaveDG2D
from ader_dg_transport.dg_2D.wave_dg_adjoint_2D import WaveAdjointDG2D
from scipy.sparse.linalg import LinearOperator, lsqr


class WaveDGOptimizer2D:

    def __init__(self, forward_solver, adjoint_solver, ic_state, target_state, nsteps, forcing_func=None):

        self.forward_solver = forward_solver
        self.adjoint_solver = adjoint_solver

        self.nsteps = nsteps
        self.ic_state = np.copy(ic_state)
        self.target_state = np.copy(target_state)
        self.forcing_func = forcing_func

        self.history_data = np.zeros((nsteps, 4) + self.forward_solver.state.shape)
        self.adjoint_data = np.zeros((nsteps, 4) + self.forward_solver.state.shape)

        self.W = self.adjoint_solver.weights_2D[None, None, None] * self.adjoint_solver.dy * self.adjoint_solver.dx * 0.25
        self.M = self.adjoint_solver.weights_2D[None, None] * self.adjoint_solver.dy * self.adjoint_solver.dx * 0.25

        self.W_half = np.sqrt(self.W)
        self.M_half = np.sqrt(self.M)

        self.W_half_inv = 1 / self.W_half
        self.M_half_inv = 1 / self.M_half

        self.G = LinearOperator(
            shape=(target_state.size, self.adjoint_solver.c.size),
            matvec=self.G_matvec,
            rmatvec=self.G_rmatvec,
            dtype=np.float64
        )

    def error(self, c_in, fill_gradient=False):

        self.forward_solver.time = 0.0
        self.forward_solver.state[:] = self.ic_state
        self.forward_solver.c[:] = c_in
        self.adjoint_solver.c[:] = self.forward_solver.c
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

        error = self.forward_solver.state - self.target_state

        return error

    def cost_function(self, c_in):
        c_in = c_in.reshape(self.forward_solver.c.shape)
        error = self.error(c_in, fill_gradient=False)
        return 0.5 * self.forward_solver.norm(*self.forward_solver.get_vars(error)) ** 2

    def jac_function(self, c_in):
        c_in = c_in.reshape(self.forward_solver.c.shape)
        error = self.error(c_in, fill_gradient=True)

        dedc = self.G_rmatvec(error * self.W_half).reshape(self.forward_solver.c.shape) * self.M_half

        return dedc.ravel()

    def G_matvec(self, dc_vec):

        dc = dc_vec.reshape(self.forward_solver.c.shape) * self.M_half_inv
        self.forward_solver.state[:] = 0.0
        for i in range(self.nsteps):
            self.forward_solver.time_step(forcing=self.history_data[i] * dc[None, None])

        return np.copy(self.forward_solver.state * self.W * self.W_half_inv).ravel()

    def G_rmatvec(self, y_vec):
        data = y_vec.reshape(self.adjoint_solver.state.shape) * self.W_half_inv

        self.adjoint_solver.state[:] = data

        for i in range(self.nsteps):
            self.adjoint_solver.time_step(stage_data=self.adjoint_data[i])

        tmp = (self.adjoint_data[::-1, ::-1] * self.history_data).sum(axis=(0, 2)) * 0.5 * self.adjoint_solver.dt
        dedc = (tmp[3] + tmp[2] * (1 / 3) + tmp[1] + tmp[0])

        return (dedc * self.M * self.M_half_inv).ravel()

    def optimization_step(self, c_in, maxiter=10):

        error = self.error(c_in, fill_gradient=True)

        b = (-error * self.W_half).ravel()

        result = lsqr(self.G, b, atol=1e-10, btol=1e-10, iter_lim=maxiter)
        dc = result[0].reshape(self.forward_solver.c.shape) * self.M_half_inv

        c_out = c_in + dc

        return c_out
