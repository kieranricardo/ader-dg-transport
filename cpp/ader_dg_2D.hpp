#pragma once
#include <stdexcept>


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
inline void ader_dg_wave_2D_kernel_element(
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
    double w,
    double bdry_flag
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
            rhs_U(a, 0, b) += bdry_flag * (0.5 * x_cfl / w) * (-H(a, 0, b) - U(a, 0, b)) * C(a, 0, b);
            rhs_H(a, 0, b) += bdry_flag * (0.5 * x_cfl / w) * (-U(a, 0, b) - H(a, 0, b)) * C(a, 0, b);

        }

    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t b = 0; b < N; ++b) {

            rhs_U(a, N-1, b) += bdry_flag * (0.5 * x_cfl / w) * (H(a, N-1, b) - U(a, N-1, b)) * C(a, N-1, b);
            rhs_H(a, N-1, b) += bdry_flag * (0.5 * x_cfl / w) * (U(a, N-1, b) - H(a, N-1, b)) * C(a, N-1, b);

        }

    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t c = 0; c < N; ++c) {
            rhs_V(a, c, 0) += bdry_flag * (0.5 * y_cfl / w) * (-H(a, c, 0) - V(a, c, 0)) * C(a, c, 0);
            rhs_H(a, c, 0) += bdry_flag * (0.5 * y_cfl / w) * (-V(a, c, 0) - H(a, c, 0)) * C(a, c, 0);
        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t c = 0; c < N; ++c) {
            rhs_V(a, c, N-1) += bdry_flag * (0.5 * y_cfl / w) * (H(a, c, N-1) - V(a, c, N-1)) * C(a, c, N-1);
            rhs_H(a, c, N-1) += bdry_flag * (0.5 * y_cfl / w) * (V(a, c, N-1) - H(a, c, N-1)) * C(a, c, N-1);
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
inline void ader_dg_wave_2D_kernel_adjoint_element(
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
    double w,
    double bdry_flag
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
            rhs_U(a, 0, b) += bdry_flag * (0.5 * x_cfl / w) * (H(a, 0, b) - U(a, 0, b)) * C(a, 0, b);
            rhs_H(a, 0, b) += bdry_flag * (0.5 * x_cfl / w) * (U(a, 0, b) - H(a, 0, b)) * C(a, 0, b);

        }

    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t b = 0; b < N; ++b) {

            rhs_U(a, N-1, b) += bdry_flag * (0.5 * x_cfl / w) * (-H(a, N-1, b) - U(a, N-1, b)) * C(a, N-1, b);
            rhs_H(a, N-1, b) += bdry_flag * (0.5 * x_cfl / w) * (-U(a, N-1, b) - H(a, N-1, b)) * C(a, N-1, b);

        }

    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t c = 0; c < N; ++c) {
            rhs_V(a, c, 0) += bdry_flag * (0.5 * y_cfl / w) * (H(a, c, 0) - V(a, c, 0)) * C(a, c, 0);
            rhs_H(a, c, 0) += bdry_flag * (0.5 * y_cfl / w) * (V(a, c, 0) - H(a, c, 0)) * C(a, c, 0);
        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        #pragma unroll
        for (std::size_t c = 0; c < N; ++c) {
            rhs_V(a, c, N-1) += bdry_flag * (0.5 * y_cfl / w) * (-H(a, c, N-1) - V(a, c, N-1)) * C(a, c, N-1);
            rhs_H(a, c, N-1) += bdry_flag * (0.5 * y_cfl / w) * (-V(a, c, N-1) - H(a, c, N-1)) * C(a, c, N-1);
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