#pragma once
#include <stdexcept>
#include "dg_matrices_2D.hpp"


template<int N>
struct ElementView2D {
    const double* __restrict__ ptr;

    inline double operator()(int a, int b) const {
        return ptr[a*N + b];
    }
};


template<int N>
struct ElementView2DMut {
    double* __restrict__ ptr;

    inline double& operator()(int a, int b) {
        return ptr[a*N + b];
    }
};


template<int N>
inline void dg_wave_2D_volume_kernel_element(
    const double* __restrict__ Uptr,
    const double* __restrict__ Vptr,
    const double* __restrict__ Hptr,
    double* __restrict__ dUdtptr,
    double* __restrict__ dVdtptr,
    double* __restrict__ dHdtptr,
    const double* __restrict__ Cptr,
    double* __restrict__ u_bufptr,
    double* __restrict__ h_bufptr,
    const double Jx,
    const double Jy,
    const double w
) {

    ElementView2D<N> u{Uptr};
    ElementView2D<N> v{Vptr};
    ElementView2D<N> h{Hptr};
    ElementView2DMut<N> dudt{dUdtptr};
    ElementView2DMut<N> dvdt{dVdtptr};
    ElementView2DMut<N> dhdt{dHdtptr};
    ElementView2D<N> c{Cptr};
    ElementView2DMut<N> u_buf{u_bufptr};
    ElementView2DMut<N> h_buf{h_bufptr};

    constexpr auto& D = DGConstants<N>::D;

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            u_buf(a, b) = u(b, a);
            h_buf(a, b) = h(b, a);
        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
                double dhdx = 0.0;
                double dudx = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdx += D[a][k] * h_buf(b, k);
                    dudx += D[a][k] * u_buf(b, k);

                }

                dudt(a, b) += - dhdx * c(a, b) / Jx;
                dhdt(a, b) += - dudx * c(a, b) / Jx;

        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
                double dhdy = 0.0;
                double dvdy = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdy += D[b][k] * h(a, k);
                    dvdy += D[b][k] * v(a, k);

                }

                dvdt(a, b) += - dhdy * c(a, b) / Jy;
                dhdt(a, b) += - dvdy * c(a, b) / Jy;

        }
    }

}


template<int N>
inline void dg_wave_adjoint_2D_volume_kernel_element(
    const double* __restrict__ Uptr,
    const double* __restrict__ Vptr,
    const double* __restrict__ Hptr,
    double* __restrict__ dUdtptr,
    double* __restrict__ dVdtptr,
    double* __restrict__ dHdtptr,
    const double* __restrict__ Cptr,
    double* __restrict__ u_bufptr,
    double* __restrict__ h_bufptr,
    const double Jx,
    const double Jy,
    const double w
) {

    ElementView2D<N> u{Uptr};
    ElementView2D<N> v{Vptr};
    ElementView2D<N> h{Hptr};
    ElementView2DMut<N> dudt{dUdtptr};
    ElementView2DMut<N> dvdt{dVdtptr};
    ElementView2DMut<N> dhdt{dHdtptr};
    ElementView2D<N> c{Cptr};
    ElementView2DMut<N> cu_buf{u_bufptr};
    ElementView2DMut<N> ch_buf{h_bufptr};

    constexpr auto& D = DGConstants<N>::D;

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
            cu_buf(a, b) = c(b, a) * u(b, a);
            ch_buf(a, b) = c(b, a) * h(b, a);
        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
                double dhdx = 0.0;
                double dudx = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdx += D[a][k] * ch_buf(b, k);
                    dudx += D[a][k] * cu_buf(b, k);
                }

                dudt(a, b) += dhdx / Jx;
                dhdt(a, b) += dudx / Jx;

        }
    }

    for (std::size_t a = 0; a < N; ++a) {
        for (std::size_t b = 0; b < N; ++b) {
                double dhdy = 0.0;
                double dvdy = 0.0;

                // Derivative in a-direction
                #pragma unroll
                for (std::size_t k = 0; k < N; ++k) {

                    dhdy += D[b][k] * h(a, k) * c(a, k);
                    dvdy += D[b][k] * v(a, k) * c(a, k);
                }

                dvdt(a, b) += dhdy / Jy;
                dhdt(a, b) += dvdy / Jy;

        }
    }

}


template<int N>
inline void dg_wave_2D_bdry_kernel_element(
    const double* __restrict__ up,
    const double* __restrict__ hp,
    const double* __restrict__ um,
    const double* __restrict__ hm,
    double* __restrict__ dudtp,
    double* __restrict__ dhdtp,
    double* __restrict__ dudtm,
    double* __restrict__ dhdtm,
    const double* __restrict__ cp,
    const double* __restrict__ cm,
    const double Jx,
    const double w
) {

    #pragma unroll
    for (std::size_t a = 0; a < N; ++a) {

        double fluxp = hp[a];
        double fluxm = hm[a];
        double num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (up[a] - um[a]);
        dudtp[a] += (num_flux - fluxp) * cp[a] / (w * Jx);
        dudtm[a] += -(num_flux - fluxm) * cm[a] / (w * Jx);

        fluxp = up[a];
        fluxm = um[a];
        num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (hp[a] - hm[a]);
        dhdtp[a] += (num_flux - fluxp) * cp[a] / (w * Jx);
        dhdtm[a] += -(num_flux - fluxm) * cm[a] / (w * Jx);


    }

}


template<int N>
inline void dg_wave_adjoint_2D_bdry_kernel_element(
    const double* __restrict__ up,
    const double* __restrict__ hp,
    const double* __restrict__ um,
    const double* __restrict__ hm,
    double* __restrict__ dudtp,
    double* __restrict__ dhdtp,
    double* __restrict__ dudtm,
    double* __restrict__ dhdtm,
    const double* __restrict__ cp,
    const double* __restrict__ cm,
    const double Jx,
    const double w
) {

    #pragma unroll
    for (std::size_t a = 0; a < N; ++a) {

        double fluxp = -cp[a] * hp[a];
        double fluxm = -cm[a] * hm[a];
        double num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (cp[a] * up[a] - cm[a] * um[a]);
        dudtp[a] += (num_flux - fluxp) / (w * Jx);
        dudtm[a] += -(num_flux - fluxm) / (w * Jx);

        fluxp = -cp[a] * up[a];
        fluxm = -cm[a] * um[a];
        num_flux = 0.5 * (fluxp + fluxm) - 0.5 * (cp[a] * hp[a] - cm[a] * hm[a]);
        dhdtp[a] += (num_flux - fluxp) / (w * Jx);
        dhdtm[a] += -(num_flux - fluxm) / (w * Jx);


    }

}
