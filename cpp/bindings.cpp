#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>

namespace py = pybind11;


template<int N>
struct ElementView {
    const double* __restrict__ ptr;

    inline double operator()(int a, int b, int c) const {
        return ptr[a*N*N + b*N + c];
    }
};


template<int N>
struct ElementViewMut {
    double* __restrict__ ptr;

    inline double& operator()(int a, int b, int c) {
        return ptr[a*N*N + b*N + c];
    }
};

template<int N>
inline void dg_volume_kernel_element(
    double* __restrict__ Uptr,
    double* __restrict__ Vptr,
    double* __restrict__ Hptr,
    const double* __restrict__ rhs_U_inptr,
    const double* __restrict__ rhs_V_inptr,
    const double* __restrict__ rhs_H_inptr,
    double* __restrict__ rhs_Uptr,
    double* __restrict__ rhs_Vptr,
    double* __restrict__ rhs_Hptr,
    const double* __restrict__ Cptr,
    const double* __restrict__ Dptr,
    const double* __restrict__ invKptr,
    double x_cfl,
    double y_cfl,
    double w
) {

    ElementViewMut<N> U{Uptr};
    ElementViewMut<N> V{Vptr};
    ElementViewMut<N> H{Hptr};
    ElementView<N> rhs_U_in{rhs_U_inptr};
    ElementView<N> rhs_V_in{rhs_V_inptr};
    ElementView<N> rhs_H_in{rhs_H_inptr};
    ElementView<N> C{Cptr};
    ElementViewMut<N> rhs_U{rhs_Uptr};
    ElementViewMut<N> rhs_V{rhs_Vptr};
    ElementViewMut<N> rhs_H{rhs_Hptr};

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {

                double dhdx = 0.0;
                double dudx = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdx += Dptr[b * N + k] * H(a, k, c);
                    dudx += Dptr[b * N + k] * U(a, k, c);
                }

                rhs_U(a, b, c) = rhs_U_in(a, b, c) - dhdx * x_cfl * C(a, b, c);
                rhs_H(a, b, c) = rhs_H_in(a, b, c) - dudx * x_cfl * C(a, b, c);
            }
        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {

                double dhdy = 0.0;
                double dvdy = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdy += Dptr[c * N + k] * H(a, b, k);
                    dvdy += Dptr[c * N + k] * V(a, b, k);
                }

                rhs_V(a, b, c) = rhs_V_in(a, b, c) - dhdy * y_cfl * C(a, b, c);
                rhs_H(a, b, c) += -dvdy * y_cfl * C(a, b, c);
            }
        }
    }

    // invert time
    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {
                double u_new = 0.0;
                double v_new = 0.0;
                double h_new = 0.0;

                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    u_new += invKptr[a * N + k] * rhs_U(k, b, c);
                    v_new += invKptr[a * N + k] * rhs_V(k, b, c);
                    h_new += invKptr[a * N + k] * rhs_H(k, b, c);

                }

                U(a, b, c) = u_new;
                V(a, b, c) = v_new;
                H(a, b, c) = h_new;

            }
        }
    }

}


template<int N>
inline void dg_kernel_element(
    double* __restrict__ Uptr,
    double* __restrict__ Vptr,
    double* __restrict__ Hptr,
    const double* __restrict__ rhs_U_inptr,
    const double* __restrict__ rhs_V_inptr,
    const double* __restrict__ rhs_H_inptr,
    double* __restrict__ rhs_Uptr,
    double* __restrict__ rhs_Vptr,
    double* __restrict__ rhs_Hptr,
    const double* __restrict__ Cptr,
    const double* __restrict__ Dptr,
    const double* __restrict__ invKptr,
    double x_cfl,
    double y_cfl,
    double w
) {

    ElementViewMut<N> U{Uptr};
    ElementViewMut<N> V{Vptr};
    ElementViewMut<N> H{Hptr};
    ElementView<N> rhs_U_in{rhs_U_inptr};
    ElementView<N> rhs_V_in{rhs_V_inptr};
    ElementView<N> rhs_H_in{rhs_H_inptr};
    ElementView<N> C{Cptr};
    ElementViewMut<N> rhs_U{rhs_Uptr};
    ElementViewMut<N> rhs_V{rhs_Vptr};
    ElementViewMut<N> rhs_H{rhs_Hptr};

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {

                double dhdx = 0.0;
                double dudx = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdx += Dptr[b * N + k] * H(a, k, c);
                    dudx += Dptr[b * N + k] * U(a, k, c);
                }

                rhs_U(a, b, c) = rhs_U_in(a, b, c) - dhdx * x_cfl * C(a, b, c);
                rhs_H(a, b, c) = rhs_H_in(a, b, c) - dudx * x_cfl * C(a, b, c);
            }
        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {

                double dhdy = 0.0;
                double dvdy = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdy += Dptr[c * N + k] * H(a, b, k);
                    dvdy += Dptr[c * N + k] * V(a, b, k);
                }

                rhs_V(a, b, c) = rhs_V_in(a, b, c) - dhdy * y_cfl * C(a, b, c);
                rhs_H(a, b, c) += -dvdy * y_cfl * C(a, b, c);
            }
        }
    }

    // boundary terms
    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t b = 0; b < N; ++b) {
            rhs_U(a, 0, b) += (0.5 * x_cfl / w) * (-H(a, 0, b) - U(a, 0, b)) * C(a, 0, b);
            rhs_H(a, 0, b) += (0.5 * x_cfl / w) * (-U(a, 0, b) - H(a, 0, b)) * C(a, 0, b);

        }

    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t b = 0; b < N; ++b) {

            rhs_U(a, N-1, b) += (0.5 * x_cfl / w) * (H(a, N-1, b) - U(a, N-1, b)) * C(a, N-1, b);
            rhs_H(a, N-1, b) += (0.5 * x_cfl / w) * (U(a, N-1, b) - H(a, N-1, b)) * C(a, N-1, b);

        }

    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t c = 0; c < N; ++c) {
            rhs_V(a, c, 0) += (0.5 * y_cfl / w) * (-H(a, c, 0) - V(a, c, 0)) * C(a, c, 0);
            rhs_H(a, c, 0) += (0.5 * y_cfl / w) * (-V(a, c, 0) - H(a, c, 0)) * C(a, c, 0);
        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t c = 0; c < N; ++c) {
            rhs_V(a, c, N-1) += (0.5 * y_cfl / w) * (H(a, c, N-1) - V(a, c, N-1)) * C(a, c, N-1);
            rhs_H(a, c, N-1) += (0.5 * y_cfl / w) * (V(a, c, N-1) - H(a, c, N-1)) * C(a, c, N-1);
        }
    }

    // invert time
    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {
                double u_new = 0.0;
                double v_new = 0.0;
                double h_new = 0.0;

                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    u_new += invKptr[a * N + k] * rhs_U(k, b, c);
                    v_new += invKptr[a * N + k] * rhs_V(k, b, c);
                    h_new += invKptr[a * N + k] * rhs_H(k, b, c);

                }

                U(a, b, c) = u_new;
                V(a, b, c) = v_new;
                H(a, b, c) = h_new;

            }
        }
    }

}


template<int N>
inline void dg_volume_kernel_adjoint_element(
    double* __restrict__ Uptr,
    double* __restrict__ Vptr,
    double* __restrict__ Hptr,
    const double* __restrict__ rhs_U_inptr,
    const double* __restrict__ rhs_V_inptr,
    const double* __restrict__ rhs_H_inptr,
    double* __restrict__ rhs_Uptr,
    double* __restrict__ rhs_Vptr,
    double* __restrict__ rhs_Hptr,
    const double* __restrict__ Cptr,
    const double* __restrict__ Dptr,
    const double* __restrict__ invKptr,
    double x_cfl,
    double y_cfl,
    double w
) {

    ElementViewMut<N> U{Uptr};
    ElementViewMut<N> V{Vptr};
    ElementViewMut<N> H{Hptr};
    ElementView<N> rhs_U_in{rhs_U_inptr};
    ElementView<N> rhs_V_in{rhs_V_inptr};
    ElementView<N> rhs_H_in{rhs_H_inptr};
    ElementView<N> C{Cptr};
    ElementViewMut<N> rhs_U{rhs_Uptr};
    ElementViewMut<N> rhs_V{rhs_Vptr};
    ElementViewMut<N> rhs_H{rhs_Hptr};

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {

                double dhdx = 0.0;
                double dudx = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdx += Dptr[b * N + k] * H(a, k, c) * C(a, k, c);
                    dudx += Dptr[b * N + k] * U(a, k, c) * C(a, k, c);
                }

                rhs_U(a, b, c) = rhs_U_in(a, b, c) + dhdx * x_cfl;
                rhs_H(a, b, c) = rhs_H_in(a, b, c) + dudx * x_cfl;
            }
        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {

                double dhdy = 0.0;
                double dvdy = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdy += Dptr[c * N + k] * H(a, b, k) * C(a, b, k);
                    dvdy += Dptr[c * N + k] * V(a, b, k) * C(a, b, k);
                }

                rhs_V(a, b, c) = rhs_V_in(a, b, c) + dhdy * y_cfl;
                rhs_H(a, b, c) += dvdy * y_cfl;
            }
        }
    }

    // invert time
    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {
                double u_new = 0.0;
                double v_new = 0.0;
                double h_new = 0.0;

                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    u_new += invKptr[a * N + k] * rhs_U(k, b, c);
                    v_new += invKptr[a * N + k] * rhs_V(k, b, c);
                    h_new += invKptr[a * N + k] * rhs_H(k, b, c);

                }

                U(a, b, c) = u_new;
                V(a, b, c) = v_new;
                H(a, b, c) = h_new;

            }
        }
    }

}


template<int N>
inline void dg_kernel_adjoint_element(
    double* __restrict__ Uptr,
    double* __restrict__ Vptr,
    double* __restrict__ Hptr,
    const double* __restrict__ rhs_U_inptr,
    const double* __restrict__ rhs_V_inptr,
    const double* __restrict__ rhs_H_inptr,
    double* __restrict__ rhs_Uptr,
    double* __restrict__ rhs_Vptr,
    double* __restrict__ rhs_Hptr,
    const double* __restrict__ Cptr,
    const double* __restrict__ Dptr,
    const double* __restrict__ invKptr,
    double x_cfl,
    double y_cfl,
    double w
) {

    ElementViewMut<N> U{Uptr};
    ElementViewMut<N> V{Vptr};
    ElementViewMut<N> H{Hptr};
    ElementView<N> rhs_U_in{rhs_U_inptr};
    ElementView<N> rhs_V_in{rhs_V_inptr};
    ElementView<N> rhs_H_in{rhs_H_inptr};
    ElementView<N> C{Cptr};
    ElementViewMut<N> rhs_U{rhs_Uptr};
    ElementViewMut<N> rhs_V{rhs_Vptr};
    ElementViewMut<N> rhs_H{rhs_Hptr};

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {

                double dhdx = 0.0;
                double dudx = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdx += Dptr[b * N + k] * H(a, k, c) * C(a, k, c);
                    dudx += Dptr[b * N + k] * U(a, k, c) * C(a, k, c);
                }

                rhs_U(a, b, c) = rhs_U_in(a, b, c) + dhdx * x_cfl;
                rhs_H(a, b, c) = rhs_H_in(a, b, c) + dudx * x_cfl;
            }
        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {

                double dhdy = 0.0;
                double dvdy = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdy += Dptr[c * N + k] * H(a, b, k) * C(a, b, k);
                    dvdy += Dptr[c * N + k] * V(a, b, k) * C(a, b, k);
                }

                rhs_V(a, b, c) = rhs_V_in(a, b, c) + dhdy * y_cfl;
                rhs_H(a, b, c) += dvdy * y_cfl;
            }
        }
    }

    // boundary terms
    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t b = 0; b < N; ++b) {
            rhs_U(a, 0, b) += (0.5 * x_cfl / w) * (H(a, 0, b) - U(a, 0, b)) * C(a, 0, b);
            rhs_H(a, 0, b) += (0.5 * x_cfl / w) * (U(a, 0, b) - H(a, 0, b)) * C(a, 0, b);

        }

    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t b = 0; b < N; ++b) {

            rhs_U(a, N-1, b) += (0.5 * x_cfl / w) * (-H(a, N-1, b) - U(a, N-1, b)) * C(a, N-1, b);
            rhs_H(a, N-1, b) += (0.5 * x_cfl / w) * (-U(a, N-1, b) - H(a, N-1, b)) * C(a, N-1, b);

        }

    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t c = 0; c < N; ++c) {
            rhs_V(a, c, 0) += (0.5 * y_cfl / w) * (H(a, c, 0) - V(a, c, 0)) * C(a, c, 0);
            rhs_H(a, c, 0) += (0.5 * y_cfl / w) * (V(a, c, 0) - H(a, c, 0)) * C(a, c, 0);
        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t c = 0; c < N; ++c) {
            rhs_V(a, c, N-1) += (0.5 * y_cfl / w) * (-H(a, c, N-1) - V(a, c, N-1)) * C(a, c, N-1);
            rhs_H(a, c, N-1) += (0.5 * y_cfl / w) * (-V(a, c, N-1) - H(a, c, N-1)) * C(a, c, N-1);
        }
    }

    // invert time
    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            for (std::size_t c = 0; c < N; ++c) {
                double u_new = 0.0;
                double v_new = 0.0;
                double h_new = 0.0;

                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    u_new += invKptr[a * N + k] * rhs_U(k, b, c);
                    v_new += invKptr[a * N + k] * rhs_V(k, b, c);
                    h_new += invKptr[a * N + k] * rhs_H(k, b, c);

                }

                U(a, b, c) = u_new;
                V(a, b, c) = v_new;
                H(a, b, c) = h_new;

            }
        }
    }

}


void dg_volume_kernel_py(
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

    auto rhs_U_in = rhs_u_in.unchecked<5>();
    auto rhs_V_in = rhs_v_in.unchecked<5>();
    auto rhs_H_in = rhs_h_in.unchecked<5>();

    auto rhs_U = rhs_u.mutable_unchecked<3>();
    auto rhs_V = rhs_v.mutable_unchecked<3>();
    auto rhs_H = rhs_h.mutable_unchecked<3>();

    // DG volume contraction
    #pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (int ii = 0; ii < maxiter; ++ii) {
                if (n == 3) {
                    dg_volume_kernel_element<3>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 4) {
                    dg_volume_kernel_element<4>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 5) {
                    dg_volume_kernel_element<5>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 6) {
                    dg_volume_kernel_element<6>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                }
            }
        }
    }
}


void  dg_kernel_py(
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

    auto rhs_U_in = rhs_u_in.unchecked<5>();
    auto rhs_V_in = rhs_v_in.unchecked<5>();
    auto rhs_H_in = rhs_h_in.unchecked<5>();

    auto rhs_U = rhs_u.mutable_unchecked<3>();
    auto rhs_V = rhs_v.mutable_unchecked<3>();
    auto rhs_H = rhs_h.mutable_unchecked<3>();

    // DG volume contraction
    #pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (int ii = 0; ii < maxiter; ++ii) {
                if (n == 3) {
                    dg_kernel_element<3>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 4) {
                    dg_kernel_element<4>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 5) {
                    dg_kernel_element<5>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 6) {
                    dg_kernel_element<6>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                }
            }
        }
    }

}


void dg_volume_kernel_adjoint_py(
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

    auto rhs_U_in = rhs_u_in.unchecked<5>();
    auto rhs_V_in = rhs_v_in.unchecked<5>();
    auto rhs_H_in = rhs_h_in.unchecked<5>();

    auto rhs_U = rhs_u.mutable_unchecked<3>();
    auto rhs_V = rhs_v.mutable_unchecked<3>();
    auto rhs_H = rhs_h.mutable_unchecked<3>();

    // DG volume contraction
    #pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (int ii = 0; ii < maxiter; ++ii) {
                if (n == 3) {
                    dg_volume_kernel_adjoint_element<3>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 4) {
                    dg_volume_kernel_adjoint_element<4>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 5) {
                    dg_volume_kernel_adjoint_element<5>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 6) {
                    dg_volume_kernel_adjoint_element<6>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                }
            }
        }
    }
}

void dg_kernel_adjoint_py(
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

    auto rhs_U_in = rhs_u_in.unchecked<5>();
    auto rhs_V_in = rhs_v_in.unchecked<5>();
    auto rhs_H_in = rhs_h_in.unchecked<5>();

    auto rhs_U = rhs_u.mutable_unchecked<3>();
    auto rhs_V = rhs_v.mutable_unchecked<3>();
    auto rhs_H = rhs_h.mutable_unchecked<3>();

    // DG volume contraction
    #pragma omp parallel for collapse(2)
    for (std::size_t i = 0; i < nx; ++i) {
        for (std::size_t j = 0; j < ny; ++j) {
            for (int ii = 0; ii < maxiter; ++ii) {
                if (n == 3) {
                    dg_kernel_adjoint_element<3>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 4) {
                    dg_kernel_adjoint_element<4>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 5) {
                    dg_kernel_adjoint_element<5>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                } else if (n == 6) {
                    dg_kernel_adjoint_element<6>(
                        &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                        &rhs_U(0, 0, 0), &rhs_V(0, 0, 0), &rhs_H(0, 0, 0), &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w
                    );
                }
            }
        }
    }

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