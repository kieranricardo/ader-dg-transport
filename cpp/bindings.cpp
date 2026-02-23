#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;

py::array_t<double> dg_volume_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> u,
    py::array_t<double, py::array::c_style | py::array::forcecast> v,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_u_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_v_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_h_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> D,
    py::array_t<double, py::array::c_style | py::array::forcecast> invK,
    double x_cfl,
    double y_cfl,
    double w,
    int maxiter,
    double tol
) {
    if (u.ndim() != 5)
        throw std::runtime_error("u must have shape (nx, ny, n, n, n)");

    const std::size_t nx = u.shape(0);
    const std::size_t ny = u.shape(1);
    const std::size_t n  = u.shape(2);

    if (u.shape(3) != n || u.shape(4) != n)
        throw std::runtime_error("Expected tensor shape (nx, ny, n, n, n)");

    if (D.ndim() != 2 || D.shape(0) != n || D.shape(1) != n)
        throw std::runtime_error("D must have shape (n, n)");

    // Create output array
    auto out = py::array_t<double>({
        (py::ssize_t)3,
        (py::ssize_t)nx,
        (py::ssize_t)ny,
        (py::ssize_t)n,
        (py::ssize_t)n,
        (py::ssize_t)n
    });

    auto rhs_u = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});
    auto rhs_v = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});
    auto rhs_h = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});

    // Fast accessors (no bounds checks)
    auto U = u.mutable_unchecked<5>();
    auto V = v.mutable_unchecked<5>();
    auto H = h.mutable_unchecked<5>();
    auto C = c_arr.unchecked<5>();
    auto Dmat = D.unchecked<2>();
    auto invKmat = invK.unchecked<2>();
    auto O = out.mutable_unchecked<6>();

    auto rhs_U_in = rhs_u_in.unchecked<5>();
    auto rhs_V_in = rhs_v_in.unchecked<5>();
    auto rhs_H_in = rhs_h_in.unchecked<5>();

    auto rhs_U = rhs_u.mutable_unchecked<3>();
    auto rhs_V = rhs_v.mutable_unchecked<3>();
    auto rhs_H = rhs_h.mutable_unchecked<3>();

    // DG volume contraction
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {

            for (int ii = 0; ii < maxiter; ++ii) {
                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t b = 0; b < n; ++b) {
                        for (std::size_t c = 0; c < n; ++c) {

                            double dhdx = 0.0;
                            double dhdy = 0.0;
                            double dudx = 0.0;
                            double dvdy = 0.0;

                            // Derivative in a-direction
                            for (std::size_t k = 0; k < n; ++k) {

                                dhdx += Dmat(b, k) * H(i, j, a, k, c);
                                dhdy += Dmat(c, k) * H(i, j, a, b, k);

                                dudx += Dmat(b, k) * U(i, j, a, k, c);
                                dvdy += Dmat(c, k) * V(i, j, a, b, k);
                            }

                            rhs_U(a, b, c) = rhs_U_in(i, j, a, b, c) - dhdx * x_cfl * C(i, j, a, b, c);
                            rhs_V(a, b, c) = rhs_V_in(i, j, a, b, c) - dhdy * y_cfl * C(i, j, a, b, c);
                            rhs_H(a, b, c) = rhs_H_in(i, j, a, b, c) - (dudx * x_cfl + dvdy * y_cfl) * C(i, j, a, b, c);
                        }
                    }
                }

                double diff = 0.0;
                // invert time
                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t b = 0; b < n; ++b) {
                        for (std::size_t c = 0; c < n; ++c) {
                            double u_new = 0.0;
                            double v_new = 0.0;
                            double h_new = 0.0;

                            for (std::size_t k = 0; k < n; ++k) {

                                u_new += invKmat(a, k) * rhs_U(k, b, c);
                                v_new += invKmat(a, k) * rhs_V(k, b, c);
                                h_new += invKmat(a, k) * rhs_H(k, b, c);

                            }

                            diff += std::abs(u_new - U(i, j, a, b, c));
                            diff += std::abs(v_new - V(i, j, a, b, c));
                            diff += std::abs(h_new - H(i, j, a, b, c));

                            U(i, j, a, b, c) = u_new;
                            V(i, j, a, b, c) = v_new;
                            H(i, j, a, b, c) = h_new;

                        }
                    }
                }

                diff = diff / (3 * n * n * n);

                if (diff < tol) {
                  break;
                }

            }

        }
    }

    return out;
}

py::array_t<double>  dg_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> u,
    py::array_t<double, py::array::c_style | py::array::forcecast> v,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_u_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_v_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_h_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> D,
    py::array_t<double, py::array::c_style | py::array::forcecast> invK,
    double x_cfl,
    double y_cfl,
    double w,
    int maxiter,
    double tol
) {
    if (u.ndim() != 5)
        throw std::runtime_error("u must have shape (nx, ny, n, n, n)");

    const std::size_t nx = u.shape(0);
    const std::size_t ny = u.shape(1);
    const std::size_t n  = u.shape(2);

    if (u.shape(3) != n || u.shape(4) != n)
        throw std::runtime_error("Expected tensor shape (nx, ny, n, n, n)");

    if (D.ndim() != 2 || D.shape(0) != n || D.shape(1) != n)
        throw std::runtime_error("D must have shape (n, n)");

    auto out = py::array_t<double>({
        (py::ssize_t)3,
        (py::ssize_t)nx,
        (py::ssize_t)ny,
        (py::ssize_t)n,
        (py::ssize_t)n,
        (py::ssize_t)n
    });

    auto rhs_u = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});
    auto rhs_v = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});
    auto rhs_h = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});

    // Fast accessors (no bounds checks)
    auto U = u.mutable_unchecked<5>();
    auto V = v.mutable_unchecked<5>();
    auto H = h.mutable_unchecked<5>();
    auto C = c_arr.unchecked<5>();
    auto Dmat = D.unchecked<2>();
    auto invKmat = invK.unchecked<2>();

    auto O = out.mutable_unchecked<6>();

    auto rhs_U_in = rhs_u_in.unchecked<5>();
    auto rhs_V_in = rhs_v_in.unchecked<5>();
    auto rhs_H_in = rhs_h_in.unchecked<5>();

    auto rhs_U = rhs_u.mutable_unchecked<3>();
    auto rhs_V = rhs_v.mutable_unchecked<3>();
    auto rhs_H = rhs_h.mutable_unchecked<3>();

    // DG volume contraction
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {

            for (int ii = 0; ii < maxiter; ++ii) {
            // each element

                // volume terms
                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t b = 0; b < n; ++b) {
                        for (std::size_t c = 0; c < n; ++c) {

                            double dhdx = 0.0;
                            double dhdy = 0.0;
                            double dudx = 0.0;
                            double dvdy = 0.0;

                            // Derivative in a-direction
                            for (std::size_t k = 0; k < n; ++k) {

                                dhdx += Dmat(b, k) * H(i, j, a, k, c);
                                dhdy += Dmat(c, k) * H(i, j, a, b, k);

                                dudx += Dmat(b, k) * U(i, j, a, k, c);
                                dvdy += Dmat(c, k) * V(i, j, a, b, k);
                            }

                            rhs_U(a, b, c) = rhs_U_in(i, j, a, b, c) - dhdx * x_cfl * C(i, j, a, b, c);
                            rhs_V(a, b, c) = rhs_V_in(i, j, a, b, c) - dhdy * y_cfl * C(i, j, a, b, c);
                            rhs_H(a, b, c) = rhs_H_in(i, j, a, b, c) - (dudx * x_cfl + dvdy * y_cfl) * C(i, j, a, b, c);
                        }
                    }
                }

                // boundary terms
                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t b = 0; b < n; ++b) {
                        rhs_U(a, 0, b) += (0.5 * x_cfl / w) * (-H(i, j, a, 0, b) - U(i, j, a, 0, b)) * C(i, j, a, 0, b);
                        rhs_H(a, 0, b) += (0.5 * x_cfl / w) * (-U(i, j, a, 0, b) - H(i, j, a, 0, b)) * C(i, j, a, 0, b);

                        rhs_U(a, n-1, b) += (0.5 * x_cfl / w) * (H(i, j, a, n-1, b) - U(i, j, a, n-1, b)) * C(i, j, a, n-1, b);
                        rhs_H(a, n-1, b) += (0.5 * x_cfl / w) * (U(i, j, a, n-1, b) - H(i, j, a, n-1, b)) * C(i, j, a, n-1, b);

                    }

                }

                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t c = 0; c < n; ++c) {
                        rhs_V(a, c, 0) += (0.5 * y_cfl / w) * (-H(i, j, a, c, 0) - V(i, j, a, c, 0)) * C(i, j, a, c, 0);
                        rhs_V(a, c, n-1) += (0.5 * y_cfl / w) * (H(i, j, a, c, n-1) - V(i, j, a, c, n-1)) * C(i, j, a, c, n-1);

                        rhs_H(a, c, 0) += (0.5 * y_cfl / w) * (-V(i, j, a, c, 0) - H(i, j, a, c, 0)) * C(i, j, a, c, 0);
                        rhs_H(a, c, n-1) += (0.5 * y_cfl / w) * (V(i, j, a, c, n-1) - H(i, j, a, c, n-1)) * C(i, j, a, c, n-1);

                    }

                }

                double diff = 0.0;
                // invert time
                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t b = 0; b < n; ++b) {
                        for (std::size_t c = 0; c < n; ++c) {
                            double u_new = 0.0;
                            double v_new = 0.0;
                            double h_new = 0.0;

                            for (std::size_t k = 0; k < n; ++k) {

                                u_new += invKmat(a, k) * rhs_U(k, b, c);
                                v_new += invKmat(a, k) * rhs_V(k, b, c);
                                h_new += invKmat(a, k) * rhs_H(k, b, c);

                            }

                            diff += std::abs(u_new - U(i, j, a, b, c));
                            diff += std::abs(v_new - V(i, j, a, b, c));
                            diff += std::abs(h_new - H(i, j, a, b, c));

                            U(i, j, a, b, c) = u_new;
                            V(i, j, a, b, c) = v_new;
                            H(i, j, a, b, c) = h_new;

                        }
                    }
                }

                diff = diff / (3 * n * n * n);

                if (diff < tol) {
                  break;
                }

            }

        }
    }

    return out;

}


py::array_t<double> dg_volume_kernel_adjoint_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> u,
    py::array_t<double, py::array::c_style | py::array::forcecast> v,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_u_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_v_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_h_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> D,
    py::array_t<double, py::array::c_style | py::array::forcecast> invK,
    double x_cfl,
    double y_cfl,
    double w,
    int maxiter,
    double tol
) {
    if (u.ndim() != 5)
        throw std::runtime_error("u must have shape (nx, ny, n, n, n)");

    const std::size_t nx = u.shape(0);
    const std::size_t ny = u.shape(1);
    const std::size_t n  = u.shape(2);

    if (u.shape(3) != n || u.shape(4) != n)
        throw std::runtime_error("Expected tensor shape (nx, ny, n, n, n)");

    if (D.ndim() != 2 || D.shape(0) != n || D.shape(1) != n)
        throw std::runtime_error("D must have shape (n, n)");

    // Create output array
    auto out = py::array_t<double>({
        (py::ssize_t)3,
        (py::ssize_t)nx,
        (py::ssize_t)ny,
        (py::ssize_t)n,
        (py::ssize_t)n,
        (py::ssize_t)n
    });

    auto rhs_u = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});
    auto rhs_v = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});
    auto rhs_h = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});

    // Fast accessors (no bounds checks)
    auto U = u.mutable_unchecked<5>();
    auto V = v.mutable_unchecked<5>();
    auto H = h.mutable_unchecked<5>();
    auto C = c_arr.unchecked<5>();
    auto Dmat = D.unchecked<2>();
    auto invKmat = invK.unchecked<2>();
    auto O = out.mutable_unchecked<6>();

    auto rhs_U_in = rhs_u_in.unchecked<5>();
    auto rhs_V_in = rhs_v_in.unchecked<5>();
    auto rhs_H_in = rhs_h_in.unchecked<5>();

    auto rhs_U = rhs_u.mutable_unchecked<3>();
    auto rhs_V = rhs_v.mutable_unchecked<3>();
    auto rhs_H = rhs_h.mutable_unchecked<3>();

    // DG volume contraction
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {

            for (int ii = 0; ii < maxiter; ++ii) {
                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t b = 0; b < n; ++b) {
                        for (std::size_t c = 0; c < n; ++c) {

                            double dhdx = 0.0;
                            double dhdy = 0.0;
                            double dudx = 0.0;
                            double dvdy = 0.0;

                            // Derivative in a-direction
                            for (std::size_t k = 0; k < n; ++k) {

                                dhdx += Dmat(b, k) * H(i, j, a, k, c) * C(i, j, a, k, c);
                                dhdy += Dmat(c, k) * H(i, j, a, b, k) * C(i, j, a, b, k);

                                dudx += Dmat(b, k) * U(i, j, a, k, c) * C(i, j, a, k, c);
                                dvdy += Dmat(c, k) * V(i, j, a, b, k) * C(i, j, a, b, k);
                            }

                            rhs_U(a, b, c) = rhs_U_in(i, j, a, b, c) - dhdx * x_cfl;
                            rhs_V(a, b, c) = rhs_V_in(i, j, a, b, c) - dhdy * y_cfl;
                            rhs_H(a, b, c) = rhs_H_in(i, j, a, b, c) - (dudx * x_cfl + dvdy * y_cfl);
                        }
                    }
                }

                double diff = 0.0;
                // invert time
                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t b = 0; b < n; ++b) {
                        for (std::size_t c = 0; c < n; ++c) {
                            double u_new = 0.0;
                            double v_new = 0.0;
                            double h_new = 0.0;

                            for (std::size_t k = 0; k < n; ++k) {

                                u_new += invKmat(a, k) * rhs_U(k, b, c);
                                v_new += invKmat(a, k) * rhs_V(k, b, c);
                                h_new += invKmat(a, k) * rhs_H(k, b, c);

                            }

                            diff += std::abs(u_new - U(i, j, a, b, c));
                            diff += std::abs(v_new - V(i, j, a, b, c));
                            diff += std::abs(h_new - H(i, j, a, b, c));

                            U(i, j, a, b, c) = u_new;
                            V(i, j, a, b, c) = v_new;
                            H(i, j, a, b, c) = h_new;

                        }
                    }
                }

                diff = diff / (3 * n * n * n);

                if (diff < tol) {
                  break;
                }

            }

        }
    }

    return out;
}

py::array_t<double>  dg_kernel_adjoint_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> u,
    py::array_t<double, py::array::c_style | py::array::forcecast> v,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_u_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_v_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> rhs_h_in,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> D,
    py::array_t<double, py::array::c_style | py::array::forcecast> invK,
    double x_cfl,
    double y_cfl,
    double w,
    int maxiter,
    double tol
) {
    if (u.ndim() != 5)
        throw std::runtime_error("u must have shape (nx, ny, n, n, n)");

    const std::size_t nx = u.shape(0);
    const std::size_t ny = u.shape(1);
    const std::size_t n  = u.shape(2);

    if (u.shape(3) != n || u.shape(4) != n)
        throw std::runtime_error("Expected tensor shape (nx, ny, n, n, n)");

    if (D.ndim() != 2 || D.shape(0) != n || D.shape(1) != n)
        throw std::runtime_error("D must have shape (n, n)");

    auto out = py::array_t<double>({
        (py::ssize_t)3,
        (py::ssize_t)nx,
        (py::ssize_t)ny,
        (py::ssize_t)n,
        (py::ssize_t)n,
        (py::ssize_t)n
    });

    auto rhs_u = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});
    auto rhs_v = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});
    auto rhs_h = py::array_t<double>({(py::ssize_t)n, (py::ssize_t)n, (py::ssize_t)n});

    // Fast accessors (no bounds checks)
    auto U = u.mutable_unchecked<5>();
    auto V = v.mutable_unchecked<5>();
    auto H = h.mutable_unchecked<5>();
    auto C = c_arr.unchecked<5>();
    auto Dmat = D.unchecked<2>();
    auto invKmat = invK.unchecked<2>();

    auto O = out.mutable_unchecked<6>();

    auto rhs_U_in = rhs_u_in.unchecked<5>();
    auto rhs_V_in = rhs_v_in.unchecked<5>();
    auto rhs_H_in = rhs_h_in.unchecked<5>();

    auto rhs_U = rhs_u.mutable_unchecked<3>();
    auto rhs_V = rhs_v.mutable_unchecked<3>();
    auto rhs_H = rhs_h.mutable_unchecked<3>();

    // DG volume contraction
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {

            for (int ii = 0; ii < maxiter; ++ii) {
            // each element

                // volume terms
                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t b = 0; b < n; ++b) {
                        for (std::size_t c = 0; c < n; ++c) {

                            double dhdx = 0.0;
                            double dhdy = 0.0;
                            double dudx = 0.0;
                            double dvdy = 0.0;

                            // Derivative in a-direction
                            for (std::size_t k = 0; k < n; ++k) {

                                dhdx += Dmat(b, k) * H(i, j, a, k, c) * C(i, j, a, k, c);
                                dhdy += Dmat(c, k) * H(i, j, a, b, k) * C(i, j, a, b, k);

                                dudx += Dmat(b, k) * U(i, j, a, k, c) * C(i, j, a, k, c);
                                dvdy += Dmat(c, k) * V(i, j, a, b, k) * C(i, j, a, b, k);
                            }

                            rhs_U(a, b, c) = rhs_U_in(i, j, a, b, c) + dhdx * x_cfl;
                            rhs_V(a, b, c) = rhs_V_in(i, j, a, b, c) + dhdy * y_cfl;
                            rhs_H(a, b, c) = rhs_H_in(i, j, a, b, c) + (dudx * x_cfl + dvdy * y_cfl);
                        }
                    }
                }

                // boundary terms
                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t b = 0; b < n; ++b) {
                        rhs_U(a, 0, b) += (0.5 * x_cfl / w) * (H(i, j, a, 0, b) - U(i, j, a, 0, b)) * C(i, j, a, 0, b);
                        rhs_H(a, 0, b) += (0.5 * x_cfl / w) * (U(i, j, a, 0, b) - H(i, j, a, 0, b)) * C(i, j, a, 0, b);

                        rhs_U(a, n-1, b) += (0.5 * x_cfl / w) * (-H(i, j, a, n-1, b) - U(i, j, a, n-1, b)) * C(i, j, a, n-1, b);
                        rhs_H(a, n-1, b) += (0.5 * x_cfl / w) * (-U(i, j, a, n-1, b) - H(i, j, a, n-1, b)) * C(i, j, a, n-1, b);

                    }

                }

                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t c = 0; c < n; ++c) {
                        rhs_V(a, c, 0) += (0.5 * y_cfl / w) * (H(i, j, a, c, 0) - V(i, j, a, c, 0)) * C(i, j, a, c, 0);
                        rhs_V(a, c, n-1) += (0.5 * y_cfl / w) * (-H(i, j, a, c, n-1) - V(i, j, a, c, n-1)) * C(i, j, a, c, n-1);

                        rhs_H(a, c, 0) += (0.5 * y_cfl / w) * (V(i, j, a, c, 0) - H(i, j, a, c, 0)) * C(i, j, a, c, 0);
                        rhs_H(a, c, n-1) += (0.5 * y_cfl / w) * (-V(i, j, a, c, n-1) - H(i, j, a, c, n-1)) * C(i, j, a, c, n-1);

                    }

                }

                double diff = 0.0;
                // invert time
                for (std::size_t a = 0; a < n; ++a) {
                    for (std::size_t b = 0; b < n; ++b) {
                        for (std::size_t c = 0; c < n; ++c) {
                            double u_new = 0.0;
                            double v_new = 0.0;
                            double h_new = 0.0;

                            for (std::size_t k = 0; k < n; ++k) {

                                u_new += invKmat(a, k) * rhs_U(k, b, c);
                                v_new += invKmat(a, k) * rhs_V(k, b, c);
                                h_new += invKmat(a, k) * rhs_H(k, b, c);

                            }

                            diff += std::abs(u_new - U(i, j, a, b, c));
                            diff += std::abs(v_new - V(i, j, a, b, c));
                            diff += std::abs(h_new - H(i, j, a, b, c));

                            U(i, j, a, b, c) = u_new;
                            V(i, j, a, b, c) = v_new;
                            H(i, j, a, b, c) = h_new;

                        }
                    }
                }

                diff = diff / (3 * n * n * n);

                if (diff < tol) {
                  break;
                }

            }

        }
    }

    return out;

}


PYBIND11_MODULE(_core, m) {
    m.doc() = "DG kernels";

    m.def("dg_volume_kernel",
          &dg_volume_kernel_py,
          py::arg("u"),
          py::arg("v"),
          py::arg("h"),
          py::arg("rhs_u"),
          py::arg("rhs_v"),
          py::arg("rhs_h"),
          py::arg("c"),
          py::arg("D"),
          py::arg("invK"),
          py::arg("x_cfl"),
          py::arg("y_cfl"),
          py::arg("w"),
          py::arg("maxiter"),
          py::arg("tol"),
          "Example DG tensor contraction in reference direction 0");

    m.def("dg_kernel",
        &dg_kernel_py,
        py::arg("u"),
        py::arg("v"),
        py::arg("h"),
        py::arg("rhs_u"),
        py::arg("rhs_v"),
        py::arg("rhs_h"),
        py::arg("c"),
        py::arg("D"),
        py::arg("invK"),
        py::arg("x_cfl"),
        py::arg("y_cfl"),
        py::arg("w"),
        py::arg("maxiter"),
        py::arg("tol"),
        "Example DG tensor contraction in reference direction 0");

        m.def("dg_volume_kernel_adjoint",
          &dg_volume_kernel_adjoint_py,
          py::arg("u"),
          py::arg("v"),
          py::arg("h"),
          py::arg("rhs_u"),
          py::arg("rhs_v"),
          py::arg("rhs_h"),
          py::arg("c"),
          py::arg("D"),
          py::arg("invK"),
          py::arg("x_cfl"),
          py::arg("y_cfl"),
          py::arg("w"),
          py::arg("maxiter"),
          py::arg("tol"),
          "Example DG tensor contraction in reference direction 0");

    m.def("dg_kernel_adjoint",
        &dg_kernel_adjoint_py,
        py::arg("u"),
        py::arg("v"),
        py::arg("h"),
        py::arg("rhs_u"),
        py::arg("rhs_v"),
        py::arg("rhs_h"),
        py::arg("c"),
        py::arg("D"),
        py::arg("invK"),
        py::arg("x_cfl"),
        py::arg("y_cfl"),
        py::arg("w"),
        py::arg("maxiter"),
        py::arg("tol"),
        "Example DG tensor contraction in reference direction 0");
}