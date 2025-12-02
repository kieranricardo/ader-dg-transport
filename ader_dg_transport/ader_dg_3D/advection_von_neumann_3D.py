import numpy as np
import scipy


def dg_norm(arr, solver):
    weights = solver.weights_3D.ravel()[None, :]

    return np.sqrt((weights * abs(arr) ** 2).sum(axis=1))


def power_iteration(mat, solver, verbose=False):
    if len(mat.shape) == 2:
        mat = mat[None]

    vec = np.random.random(mat.shape[:-1])
    norm = dg_norm(vec, solver)
    vec /= norm[:, None]
    growth = np.zeros_like(norm)

    for i in range(100):
        vec = np.einsum('abc,ac->ab', mat, vec)
        norm = dg_norm(vec, solver)
        vec /= norm[:, None]

        eig_est = norm

        growth = np.maximum(growth, norm)

    return growth


def get_vn_matrices(poly_order):
    n = poly_order + 1

    shape = (n,) * 8

    xp_to_xm = np.zeros(shape)
    yp_to_ym = np.zeros(shape)
    zp_to_zm = np.zeros(shape)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                xp_to_xm[i, 0, j, k, i, -1, j, k] = 1.0
                yp_to_ym[i, j, 0, k, i, j, -1, k] = 1.0
                zp_to_zm[i, j, k, 0, i, j, k, -1] = 1.0

    assert np.allclose(xp_to_xm, yp_to_ym.swapaxes(1, 2).swapaxes(5, 6))
    assert np.allclose(xp_to_xm, zp_to_zm.swapaxes(1, 3).swapaxes(5, 7))

    xp_to_xm = xp_to_xm.reshape((n ** 4, n ** 4))
    yp_to_ym = yp_to_ym.reshape((n ** 4, n ** 4))
    zp_to_zm = zp_to_zm.reshape((n ** 4, n ** 4))

    shape = (n,) * 7
    to_st_first = np.zeros(shape)
    to_st_last = np.zeros(shape)

    from_st_first = np.zeros(shape)
    from_st_last = np.zeros(shape)

    for i in range(n):
        for j in range(n):
            for k in range(n):
                to_st_first[0, i, j, k, i, j, k] = 1.0
                to_st_last[-1, i, j, k, i, j, k] = 1.0

                from_st_first[i, j, k, 0, i, j, k] = 1.0
                from_st_last[i, j, k, -1, i, j, k] = 1.0

    to_st_first = to_st_first.reshape((n ** 4, n ** 3))
    to_st_last = to_st_last.reshape((n ** 4, n ** 3))

    from_st_first = from_st_first.reshape((n ** 3, n ** 4))
    from_st_last = from_st_last.reshape((n ** 3, n ** 4))

    assert np.allclose(from_st_first @ to_st_first, np.eye(n ** 3))
    assert np.allclose(from_st_last @ to_st_last, np.eye(n ** 3))

    assert np.allclose(from_st_first @ to_st_last, 0.0)
    assert np.allclose(from_st_last @ to_st_first, 0.0)

    return to_st_first, to_st_last, from_st_first, from_st_last, xp_to_xm, yp_to_ym, zp_to_zm


def von_neumann_3D(solver, x_cfl, y_cfl, z_cfl, x_shifts, y_shifts, z_shifts, use_power_iteration=True, verbose=False, batch_size=100):
    import time

    t0 = time.time()
    to_st_first, to_st_last, from_st_first, from_st_last, xp_to_xm, yp_to_ym, zp_to_zm = get_vn_matrices(solver.poly_order)

    (Dt, Dx, Dy, Dz, first_space_integral,
     last_space_integral, xm_integral, xp_integral,
     ym_integral, yp_integral, zm_integral, zp_integral) = solver.get_matrices()

    t1 = time.time()
    if verbose:
        print('Setup time:', t1 - t0)
    # matrices
    M1 = Dt + x_cfl * Dx + y_cfl * Dy + z_cfl * Dz + first_space_integral

    M = M1 + x_cfl * (xm_integral - xp_integral) + y_cfl * (ym_integral - yp_integral) + z_cfl * (zm_integral - zp_integral)

    Mx = M1 + x_cfl * (xm_integral - xp_integral)
    My = M1 + y_cfl * (ym_integral - yp_integral)
    Mz = M1 + z_cfl * (zm_integral - zp_integral)

    Mxy = Mx + y_cfl * (ym_integral - yp_integral)
    Mxz = Mx + z_cfl * (zm_integral - zp_integral)
    Myz = My + z_cfl * (zm_integral - zp_integral)

    ####
    Rt = first_space_integral @ to_st_first

    xm = (x_cfl * xm_integral @ xp_to_xm)
    xp = x_cfl * xp_integral

    ym = (y_cfl * ym_integral @ yp_to_ym)
    yp = y_cfl * yp_integral

    zm = (z_cfl * zm_integral @ zp_to_zm)
    zp = z_cfl * zp_integral

    t2 = time.time()

    if verbose:
        print('Matrix construction:', t2 - t1)

    t1 = time.time()
    M1_inv = np.linalg.inv(M1)

    M_inv = np.linalg.inv(M)
    Mx_inv = np.linalg.inv(Mx)
    My_inv = np.linalg.inv(My)
    Mz_inv = np.linalg.inv(Mz)

    Mxy_inv = np.linalg.inv(Mxy)
    Mxz_inv = np.linalg.inv(Mxz)
    Myz_inv = np.linalg.inv(Myz)

    t2 = time.time()

    if verbose:
        print('Matrix inverse:', t2 - t1)


    #### first stage
    u_pred_1 = M1_inv @ Rt

    t3 = time.time()

    if verbose:
        print('First stage:', t3 - t2)

    ### second stage

    u_pred_x_0 = Mx_inv @ (Rt)
    u_pred_x_0 -= Mx_inv @ (xp @ u_pred_1)
    u_pred_x_x = Mx_inv @ (xm @ u_pred_1)

    u_pred_y_0 = My_inv @ (Rt)
    u_pred_y_0 -= My_inv @ (yp @ u_pred_1)
    u_pred_y_y = My_inv @ (ym @ u_pred_1)

    u_pred_z_0 = Mz_inv @ (Rt)
    u_pred_z_0 -= Mz_inv @ (zp @ u_pred_1)
    u_pred_z_z = Mz_inv @ (zm @ u_pred_1)

    t4 = time.time()

    if verbose:
        print('Second stage:', t4 - t3)

    ### third stage

    u_pred_xy_0 = Mxy_inv @ (Rt)

    u_pred_xy_0 -= Mxy_inv @ (xp @ u_pred_y_0)
    u_pred_xy_y = -Mxy_inv @ (xp @ u_pred_y_y)
    u_pred_xy_x = Mxy_inv @ (xm @ u_pred_y_0)
    u_pred_xy_xy = Mxy_inv @ (xm @ u_pred_y_y)

    u_pred_xy_0 -= Mxy_inv @ (yp @ u_pred_x_0)
    u_pred_xy_x -= Mxy_inv @ (yp @ u_pred_x_x)
    u_pred_xy_y += Mxy_inv @ (ym @ u_pred_x_0)
    u_pred_xy_xy += Mxy_inv @ (ym @ u_pred_x_x)

    # u_pred_xy = u_pred_xy_0 * broadcast + u_pred_xy_x * exp_x_shifts + u_pred_xy_y * exp_y_shifts
    # u_pred_xy += u_pred_xy_xy * exp_xy_shifts

    ##
    u_pred_xz_0 = Mxz_inv @ (Rt)

    u_pred_xz_0 -= Mxz_inv @ (xp @ u_pred_z_0)
    u_pred_xz_z = -Mxz_inv @ (xp @ u_pred_z_z)
    u_pred_xz_x = Mxz_inv @ (xm @ u_pred_z_0)
    u_pred_xz_xz = Mxz_inv @ (xm @ u_pred_z_z)

    u_pred_xz_0 -= Mxz_inv @ (zp @ u_pred_x_0)
    u_pred_xz_x -= Mxz_inv @ (zp @ u_pred_x_x)
    u_pred_xz_z += Mxz_inv @ (zm @ u_pred_x_0)
    u_pred_xz_xz += Mxz_inv @ (zm @ u_pred_x_x)

    # u_pred_xz = u_pred_xz_0 * broadcast + u_pred_xz_x * exp_x_shifts + u_pred_xz_z * exp_z_shifts
    # u_pred_xz += u_pred_xz_xz * exp_xz_shifts

    ##
    u_pred_yz_0 = Myz_inv @ (Rt)

    u_pred_yz_0 -= Myz_inv @ (yp @ u_pred_z_0)
    u_pred_yz_z = -Myz_inv @ (yp @ u_pred_z_z)
    u_pred_yz_y = Myz_inv @ (ym @ u_pred_z_0)
    u_pred_yz_yz = Myz_inv @ (ym @ u_pred_z_z)

    u_pred_yz_0 -= Myz_inv @ (zp @ u_pred_y_0)
    u_pred_yz_y -= Myz_inv @ (zp @ u_pred_y_y)
    u_pred_yz_z += Myz_inv @ (zm @ u_pred_y_0)
    u_pred_yz_yz += Myz_inv @ (zm @ u_pred_y_y)

    # u_pred_yz = u_pred_yz_0 * broadcast + u_pred_yz_y * exp_y_shifts + u_pred_yz_z * exp_z_shifts
    # u_pred_yz += u_pred_yz_yz * exp_yz_shifts


    t5 = time.time()

    if verbose:
        print('Third stage:', t5 - t4)

    #### corrector
    M_inv = from_st_last @ M_inv

    u_out_0 = M_inv @ Rt

    ##
    u_out_0 -= M_inv @ (xp @ u_pred_yz_0)
    u_out_y = -M_inv @ (xp @ u_pred_yz_y)
    u_out_z = -M_inv @ (xp @ u_pred_yz_z)
    u_out_yz = - M_inv @ (xp @ u_pred_yz_yz)

    u_out_x = M_inv @ (xm @ u_pred_yz_0)
    u_out_xy = M_inv @ (xm @ u_pred_yz_y)
    u_out_xz = M_inv @ (xm @ u_pred_yz_z)
    u_out_xyz = M_inv @ (xm @ u_pred_yz_yz)

    ###
    u_out_0 -= M_inv @ (yp @ u_pred_xz_0)
    u_out_x -= M_inv @ (yp @ u_pred_xz_x)
    u_out_z -= M_inv @ (yp @ u_pred_xz_z)
    u_out_xz -= M_inv @ (yp @ u_pred_xz_xz)

    u_out_y += M_inv @ (ym @ u_pred_xz_0)
    u_out_xy += M_inv @ (ym @ u_pred_xz_x)
    u_out_yz += M_inv @ (ym @ u_pred_xz_z)
    u_out_xyz += M_inv @ (ym @ u_pred_xz_xz)

    ###
    u_out_0 -= M_inv @ (zp @ u_pred_xy_0)
    u_out_x -= M_inv @ (zp @ u_pred_xy_x)
    u_out_y -= M_inv @ (zp @ u_pred_xy_y)
    u_out_xy -= M_inv @ (zp @ u_pred_xy_xy)

    u_out_z += M_inv @ (zm @ u_pred_xy_0)
    u_out_xz += M_inv @ (zm @ u_pred_xy_x)
    u_out_yz += M_inv @ (zm @ u_pred_xy_y)
    u_out_xyz += M_inv @ (zm @ u_pred_xy_xy)

    # u_out_0 = from_st_last @ u_out_0 #@ to_st_first
    #
    # u_out_x = from_st_last @ u_out_x #@ to_st_first
    # u_out_y = from_st_last @ u_out_y #@ to_st_first
    # u_out_z = from_st_last @ u_out_z #@ to_st_first
    #
    # u_out_xy = from_st_last @ u_out_xy #@ to_st_first
    # u_out_xz = from_st_last @ u_out_xz #@ to_st_first
    # u_out_yz = from_st_last @ u_out_yz #@ to_st_first
    #
    # u_out_xyz = from_st_last @ u_out_xyz #@ to_st_first

    t6 = time.time()

    if verbose:
        print('Corrector stage:', t6 - t5)
        print('All stages:', t6 - t2)


    eigs = np.zeros(x_shifts.size)

    bc_time = 0
    eig_time = 0

    for i in range(0, x_shifts.size, batch_size):

        j = min(i + batch_size, x_shifts.size)

        t0 = time.time()
        broadcast = np.ones(j - i)[:, None, None] * (1.0 + 0.0j)

        exp_x_shifts = np.exp(-x_shifts.ravel()[i:j, None, None])
        exp_y_shifts = np.exp(-y_shifts.ravel()[i:j, None, None])
        exp_z_shifts = np.exp(-z_shifts.ravel()[i:j, None, None])

        exp_xy_shifts = exp_x_shifts * exp_y_shifts
        exp_xz_shifts = exp_x_shifts * exp_z_shifts
        exp_yz_shifts = exp_y_shifts * exp_z_shifts

        u_out = u_out_0 * broadcast
        u_out += u_out_x * exp_x_shifts + u_out_y * exp_y_shifts + u_out_z * exp_z_shifts
        u_out += u_out_xy * exp_xy_shifts + u_out_xz * exp_xz_shifts + u_out_yz * exp_yz_shifts
        u_out += u_out_xyz * exp_xy_shifts * exp_z_shifts

        t1 = time.time()

        if use_power_iteration:
            eigs_ = power_iteration(u_out, solver)
        else:
            eigs_ = abs(np.linalg.eigvals(u_out)).max(axis=1)

        eigs[i:j] = eigs_

        t2 = time.time()

        bc_time += t1 - t0
        eig_time += t2 - t1

    if verbose:
        print('broadcast time:', bc_time)

    if verbose:
        print('eigs:', eig_time)

    return eigs


def matrix_von_neumann_3D(solver, x_cfl, y_cfl, z_cfl, x_shifts, y_shifts, z_shifts, use_power_iteration=True, verbose=False, batch_size=100):
    import time

    t0 = time.time()
    to_st_first, to_st_last, from_st_first, from_st_last, xp_to_xm, yp_to_ym, zp_to_zm = get_vn_matrices(solver.poly_order)

    (Dt, Dx, Dy, Dz, first_space_integral,
     last_space_integral, xm_integral, xp_integral,
     ym_integral, yp_integral, zm_integral, zp_integral) = solver.get_matrices()

    t1 = time.time()
    if verbose:
        print('Setup time:', t1 - t0)
    # matrices
    M1 = Dt + x_cfl * Dx + y_cfl * Dy + z_cfl * Dz + first_space_integral
    M1_inv = np.linalg.inv(M1)

    M = M1 + x_cfl * (xm_integral - xp_integral) + y_cfl * (ym_integral - yp_integral) + z_cfl * (zm_integral - zp_integral)
    
    Mx = M1 + x_cfl * (xm_integral - xp_integral)
    My = M1 + y_cfl * (ym_integral - yp_integral)
    Mz = M1 + z_cfl * (zm_integral - zp_integral)

    Mxy = Mx + y_cfl * (ym_integral - yp_integral)
    Mxz = Mx + z_cfl * (zm_integral - zp_integral)
    Myz = My + z_cfl * (zm_integral - zp_integral)

    ####
    Rt = first_space_integral @ to_st_first

    xm = (x_cfl * xm_integral @ xp_to_xm)
    xp = x_cfl * xp_integral

    ym = (y_cfl * ym_integral @ yp_to_ym)
    yp = y_cfl * yp_integral

    zm = (z_cfl * zm_integral @ zp_to_zm)
    zp = z_cfl * zp_integral

    t2 = time.time()

    if verbose:
        print('Matrix construction:', t2 - t1)

    t1 = time.time()
    M_inv = np.linalg.inv(M)
    Mx_inv = np.linalg.inv(Mx)
    My_inv = np.linalg.inv(My)
    Mz_inv = np.linalg.inv(Mz)

    Mxy_inv = np.linalg.inv(Mxy)
    Mxz_inv = np.linalg.inv(Mxz)
    Myz_inv = np.linalg.inv(Myz)

    t2 = time.time()

    if verbose:
        print('Matrix inverse:', t2 - t1)

    #### first stagge
    u_pred_1 = M1_inv @ Rt

    t3 = time.time()

    if verbose:
        print('First stage:', t3 - t2)

    ### second stage

    u_pred_x_0 = Mx_inv @ (Rt)
    u_pred_x_0 -= Mx_inv @ (xp @ u_pred_1)
    u_pred_x_x = (Mx_inv @ xm @ u_pred_1)

    u_pred_y_0 = My_inv @ (Rt)
    u_pred_y_0 -= My_inv @ (yp @ u_pred_1)
    u_pred_y_y = (My_inv @ ym @ u_pred_1)

    u_pred_z_0 = Mz_inv @ (Rt)
    u_pred_z_0 -= Mz_inv @ (zp @ u_pred_1)
    u_pred_z_z = (Mz_inv @ zm @ u_pred_1)

    t4 = time.time()

    if verbose:
        print('Second stage:', t4 - t3)

    ### third stage

    u_pred_xy_0 = Mxy_inv @ (Rt)

    u_pred_xy_0 -= Mxy_inv @ (xp @ u_pred_y_0)
    u_pred_xy_y = -Mxy_inv @ (xp @ u_pred_y_y)
    u_pred_xy_x = Mxy_inv @ (xm @ u_pred_y_0)
    u_pred_xy_xy = Mxy_inv @ (xm @ u_pred_y_y)

    u_pred_xy_0 -= Mxy_inv @ (yp @ u_pred_x_0)
    u_pred_xy_x -= Mxy_inv @ (yp @ u_pred_x_x)
    u_pred_xy_y += Mxy_inv @ (ym @ u_pred_x_0)
    u_pred_xy_xy += Mxy_inv @ (ym @ u_pred_x_x)

    # u_pred_xy = u_pred_xy_0 * broadcast + u_pred_xy_x * exp_x_shifts + u_pred_xy_y * exp_y_shifts
    # u_pred_xy += u_pred_xy_xy * exp_xy_shifts

    ##
    u_pred_xz_0 = Mxz_inv @ (Rt)

    u_pred_xz_0 -= Mxz_inv @ (xp @ u_pred_z_0)
    u_pred_xz_z = -Mxz_inv @ (xp @ u_pred_z_z)
    u_pred_xz_x = Mxz_inv @ (xm @ u_pred_z_0)
    u_pred_xz_xz = Mxz_inv @ (xm @ u_pred_z_z)

    u_pred_xz_0 -= Mxz_inv @ (zp @ u_pred_x_0)
    u_pred_xz_x -= Mxz_inv @ (zp @ u_pred_x_x)
    u_pred_xz_z += Mxz_inv @ (zm @ u_pred_x_0)
    u_pred_xz_xz += Mxz_inv @ (zm @ u_pred_x_x)

    # u_pred_xz = u_pred_xz_0 * broadcast + u_pred_xz_x * exp_x_shifts + u_pred_xz_z * exp_z_shifts
    # u_pred_xz += u_pred_xz_xz * exp_xz_shifts

    ##
    u_pred_yz_0 = Myz_inv @ (Rt)

    u_pred_yz_0 -= Myz_inv @ (yp @ u_pred_z_0)
    u_pred_yz_z = -Myz_inv @ (yp @ u_pred_z_z)
    u_pred_yz_y = Myz_inv @ (ym @ u_pred_z_0)
    u_pred_yz_yz = Myz_inv @ (ym @ u_pred_z_z)

    u_pred_yz_0 -= Myz_inv @ (zp @ u_pred_y_0)
    u_pred_yz_y -= Myz_inv @ (zp @ u_pred_y_y)
    u_pred_yz_z += Myz_inv @ (zm @ u_pred_y_0)
    u_pred_yz_yz += Myz_inv @ (zm @ u_pred_y_y)

    # u_pred_yz = u_pred_yz_0 * broadcast + u_pred_yz_y * exp_y_shifts + u_pred_yz_z * exp_z_shifts
    # u_pred_yz += u_pred_yz_yz * exp_yz_shifts


    t5 = time.time()

    if verbose:
        print('Third stage:', t5 - t4)

    #### corrector
    M_inv = from_st_last @ M_inv

    u_out_0 = M_inv @ Rt

    ##
    u_out_0 -= M_inv @ (xp @ u_pred_yz_0)
    u_out_y = -M_inv @ (xp @ u_pred_yz_y)
    u_out_z = -M_inv @ (xp @ u_pred_yz_z)
    u_out_yz = - M_inv @ (xp @ u_pred_yz_yz)

    u_out_x = M_inv @ (xm @ u_pred_yz_0)
    u_out_xy = M_inv @ (xm @ u_pred_yz_y)
    u_out_xz = M_inv @ (xm @ u_pred_yz_z)
    u_out_xyz = M_inv @ (xm @ u_pred_yz_yz)

    ###
    u_out_0 -= M_inv @ (yp @ u_pred_xz_0)
    u_out_x -= M_inv @ (yp @ u_pred_xz_x)
    u_out_z -= M_inv @ (yp @ u_pred_xz_z)
    u_out_xz -= M_inv @ (yp @ u_pred_xz_xz)

    u_out_y += M_inv @ (ym @ u_pred_xz_0)
    u_out_xy += M_inv @ (ym @ u_pred_xz_x)
    u_out_yz += M_inv @ (ym @ u_pred_xz_z)
    u_out_xyz += M_inv @ (ym @ u_pred_xz_xz)

    ###
    u_out_0 -= M_inv @ (zp @ u_pred_xy_0)
    u_out_x -= M_inv @ (zp @ u_pred_xy_x)
    u_out_y -= M_inv @ (zp @ u_pred_xy_y)
    u_out_xy -= M_inv @ (zp @ u_pred_xy_xy)

    u_out_z += M_inv @ (zm @ u_pred_xy_0)
    u_out_xz += M_inv @ (zm @ u_pred_xy_x)
    u_out_yz += M_inv @ (zm @ u_pred_xy_y)
    u_out_xyz += M_inv @ (zm @ u_pred_xy_xy)

    # u_out_0 = from_st_last @ u_out_0 #@ to_st_first
    #
    # u_out_x = from_st_last @ u_out_x #@ to_st_first
    # u_out_y = from_st_last @ u_out_y #@ to_st_first
    # u_out_z = from_st_last @ u_out_z #@ to_st_first
    #
    # u_out_xy = from_st_last @ u_out_xy #@ to_st_first
    # u_out_xz = from_st_last @ u_out_xz #@ to_st_first
    # u_out_yz = from_st_last @ u_out_yz #@ to_st_first
    #
    # u_out_xyz = from_st_last @ u_out_xyz #@ to_st_first

    t6 = time.time()

    if verbose:
        print('Corrector stage:', t6 - t5)
        print('All stages:', t6 - t2)


    eigs = np.zeros(x_shifts.size)

    bc_time = 0
    eig_time = 0

    batch_size = x_shifts.size
    for i in range(0, x_shifts.size, batch_size):

        j = min(i + batch_size, x_shifts.size)

        t0 = time.time()
        broadcast = np.ones(j - i)[:, None, None] * (1.0 + 0.0j)

        exp_x_shifts = np.exp(-x_shifts.ravel()[i:j, None, None])
        exp_y_shifts = np.exp(-y_shifts.ravel()[i:j, None, None])
        exp_z_shifts = np.exp(-z_shifts.ravel()[i:j, None, None])

        exp_xy_shifts = exp_x_shifts * exp_y_shifts
        exp_xz_shifts = exp_x_shifts * exp_z_shifts
        exp_yz_shifts = exp_y_shifts * exp_z_shifts

        u_out = u_out_0 * broadcast
        u_out += u_out_x * exp_x_shifts + u_out_y * exp_y_shifts + u_out_z * exp_z_shifts
        u_out += u_out_xy * exp_xy_shifts + u_out_xz * exp_xz_shifts + u_out_yz * exp_yz_shifts
        u_out += u_out_xyz * exp_xy_shifts * exp_z_shifts

        t1 = time.time()

        bc_time += t1 - t0

    if verbose:
        print('broadcast time:', bc_time)


    return u_out