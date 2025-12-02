import numpy as np
from ader_dg_transport.utils import gll, lagrange1st
from matplotlib import pyplot as plt
from scipy.linalg import lu_factor, lu_solve
import scipy
import time
import os


plot_dir = f'plots'


def get_matrices(order):
    n = order + 1
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

    xs, w_x = gll(order, iterative=True)
    D = lagrange1st(order, xs).transpose()

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

        left_time_integral[i, 0, i, 0] = 1 / w_x[-1]
        right_time_integral[i, -1, i, -1] = 1 / w_x[-1]

        first_space_integral[0, i, 0, i] = 1 / w_x[-1]
        last_space_integral[-1, i, -1, i] = 1 / w_x[-1]

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

cfls = [1e-8,] + list(np.linspace(0.01, 1.00, 100))

cmap = plt.get_cmap('tab20')
for poly_order in range(3, 15):
    _, weights = gll(poly_order, iterative=True)

    (to_st_first, to_st_last, from_st_first, from_st_last, Dx, Dt,
     volume_integral, left_time_integral, right_time_integral, first_space_integral, last_space_integral,
     pick_x0_t1, right_to_left, last_to_first
     ) = get_matrices(poly_order)

    n = poly_order + 1
    to_st_right = np.zeros((n, n, n))
    to_st_left = np.zeros((n, n, n))
    for i in range(n):
        to_st_left[i, 0, i] = 1.0
        to_st_right[i, -1, i] = 1.0

    to_st_right = to_st_right.reshape((n**2, n))
    to_st_left = to_st_left.reshape(( n**2, n))

    norms = []

    W = np.diag(weights)
    W1 = np.diag(weights ** 0.5)
    W2 = np.diag(weights ** -0.5)

    inv_volume_integral = np.diag(1 / np.diag(volume_integral))

    for cfl in cfls:

        M = (Dt + cfl * Dx) + first_space_integral
        M += cfl * (left_time_integral - right_time_integral)
        A = np.sqrt(cfl) * from_st_last @ scipy.linalg.solve(M, to_st_left / weights[-1])
        A_norm = scipy.linalg.norm(W2 @ A.T @ W @ A @ W2, ord=2)

        norms.append(A_norm)

    plt.plot(cfls, norms, label=f'Order {poly_order}', color=cmap(poly_order - 3))

plt.ylabel("$||A||_w$")
plt.xlabel("CFL")
plt.legend()
plt.grid()
plt.savefig(os.path.join(plot_dir, "new-ader-dg-A-norm.png"))
plt.show()