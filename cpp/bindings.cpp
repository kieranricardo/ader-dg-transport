#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include <omp.h>
#include "ader_dg_2D.hpp"
#include "dg_2D.hpp"

namespace py = pybind11;


void  ader_dg_wave_2D_kernel_py(
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
    double tol,
    double bdry_flag
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

    // DG volume contraction
    #pragma omp parallel
    {
        double rhs_U[n*n*n];
        double rhs_V[n*n*n];
        double rhs_H[n*n*n];

        #pragma omp for collapse(2)
        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {

                for (int ii = 0; ii < maxiter; ++ii) {
                    if (n == 3) {
                        ader_dg_wave_2D_kernel_element<3>(
                            &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                            rhs_U, rhs_V, rhs_H, &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w, bdry_flag
                        );
                    } else if (n == 4) {
                        ader_dg_wave_2D_kernel_element<4>(
                            &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                            rhs_U, rhs_V, rhs_H, &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w, bdry_flag
                        );
                    } else if (n == 5) {
                        ader_dg_wave_2D_kernel_element<5>(
                            &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                            rhs_U, rhs_V, rhs_H, &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w, bdry_flag
                        );
                    } else if (n == 6) {
                        ader_dg_wave_2D_kernel_element<6>(
                            &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                            rhs_U, rhs_V, rhs_H, &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w, bdry_flag
                        );
                    }
                }
            }
        }
    }
}


void ader_dg_wave_2D_kernel_adjoint_py(
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
    double tol,
    double bdry_flag
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

    // DG volume contraction
    #pragma omp parallel
    {
        double rhs_U[n*n*n];
        double rhs_V[n*n*n];
        double rhs_H[n*n*n];

        #pragma omp for collapse(2)
        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {

                for (int ii = 0; ii < maxiter; ++ii) {
                    if (n == 3) {
                        ader_dg_wave_2D_kernel_adjoint_element<3>(
                            &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                            rhs_U, rhs_V, rhs_H, &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w, bdry_flag
                        );
                    } else if (n == 4) {
                        ader_dg_wave_2D_kernel_adjoint_element<4>(
                            &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                            rhs_U, rhs_V, rhs_H, &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w, bdry_flag
                        );
                    } else if (n == 5) {
                        ader_dg_wave_2D_kernel_adjoint_element<5>(
                            &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                            rhs_U, rhs_V, rhs_H, &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w, bdry_flag
                        );
                    } else if (n == 6) {
                        ader_dg_wave_2D_kernel_adjoint_element<6>(
                            &U(i, j, 0, 0, 0), &V(i, j, 0, 0, 0), &H(i, j, 0, 0, 0), &rhs_U_in(i, j, 0, 0, 0), &rhs_V_in(i, j, 0, 0, 0), &rhs_H_in(i, j, 0, 0, 0),
                            rhs_U, rhs_V, rhs_H, &C(i, j, 0, 0, 0), &Dmat(0, 0), &invKmat(0, 0), x_cfl, y_cfl, w, bdry_flag
                        );
                    }
                }
            }
        }
    }

}


void  dg_wave_2D_volume_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> u,
    py::array_t<double, py::array::c_style | py::array::forcecast> v,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    py::array_t<double, py::array::c_style | py::array::forcecast> dudt,
    py::array_t<double, py::array::c_style | py::array::forcecast> dvdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> dhdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> D,
    double Jx,
    double Jy,
    double w
) {
    if (u.ndim() != 4)
        throw std::runtime_error("u must have shape (nx, ny, n, n)");

    const std::size_t nx = u.shape(0);
    const std::size_t ny = u.shape(1);
    const std::size_t n  = u.shape(2);

    if (u.shape(3) != n)
        throw std::runtime_error("Expected tensor shape (nx, ny, n, n)");

    if (D.ndim() != 2 || D.shape(0) != n || D.shape(1) != n)
        throw std::runtime_error("D must have shape (n, n)");

    // Fast accessors (no bounds checks)
    auto U = u.unchecked<4>();
    auto V = v.unchecked<4>();
    auto H = h.unchecked<4>();
    auto dUdt = dudt.mutable_unchecked<4>();
    auto dVdt = dvdt.mutable_unchecked<4>();
    auto dHdt = dhdt.mutable_unchecked<4>();
    auto C = c_arr.unchecked<4>();
    auto Dmat = D.unchecked<2>();

    // DG volume contraction
    #pragma omp parallel
    {

        #pragma omp for collapse(2)
        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {

                if (n == 3) {
                    dg_wave_2D_volume_kernel_element<3>(
                        &U(i, j, 0, 0), &V(i, j, 0, 0), &H(i, j, 0, 0),
                        &dUdt(i, j, 0, 0), &dVdt(i, j, 0, 0), &dHdt(i, j, 0, 0),
                        &C(i, j, 0, 0), &Dmat(0, 0), Jx, Jy, w
                    );
                } else if (n == 4) {
                    dg_wave_2D_volume_kernel_element<4>(
                        &U(i, j, 0, 0), &V(i, j, 0, 0), &H(i, j, 0, 0),
                        &dUdt(i, j, 0, 0), &dVdt(i, j, 0, 0), &dHdt(i, j, 0, 0),
                        &C(i, j, 0, 0), &Dmat(0, 0), Jx, Jy, w
                    );
                } else if (n == 5) {
                    dg_wave_2D_volume_kernel_element<5>(
                        &U(i, j, 0, 0), &V(i, j, 0, 0), &H(i, j, 0, 0),
                        &dUdt(i, j, 0, 0), &dVdt(i, j, 0, 0), &dHdt(i, j, 0, 0),
                        &C(i, j, 0, 0), &Dmat(0, 0), Jx, Jy, w
                    );
                } else if (n == 6) {
                    dg_wave_2D_volume_kernel_element<6>(
                        &U(i, j, 0, 0), &V(i, j, 0, 0), &H(i, j, 0, 0),
                        &dUdt(i, j, 0, 0), &dVdt(i, j, 0, 0), &dHdt(i, j, 0, 0),
                        &C(i, j, 0, 0), &Dmat(0, 0), Jx, Jy, w
                    );
                }

            }
        }
    }
}


void  dg_wave_2D_bdry_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> up,
    py::array_t<double, py::array::c_style | py::array::forcecast> hp,
    py::array_t<double, py::array::c_style | py::array::forcecast> um,
    py::array_t<double, py::array::c_style | py::array::forcecast> hm,
    py::array_t<double, py::array::c_style | py::array::forcecast> dupdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> dhpdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> dumdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> dhmdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> cp,
    py::array_t<double, py::array::c_style | py::array::forcecast> cm,
    double Jx,
    double w
) {

    // Fast accessors (no bounds checks)
    auto Up = up.unchecked<2>();
    auto Hp = hp.unchecked<2>();
    auto Um = um.unchecked<2>();
    auto Hm = hm.unchecked<2>();

    auto dUpdt = dupdt.mutable_unchecked<2>();
    auto dHpdt = dhpdt.mutable_unchecked<2>();
    auto dUmdt = dumdt.mutable_unchecked<2>();
    auto dHmdt = dhmdt.mutable_unchecked<2>();

    auto Cp = cp.unchecked<2>();
    auto Cm = cm.unchecked<2>();

    const std::size_t nx = up.shape(0);
    const std::size_t n  = up.shape(1);

    // DG bdry contraction
    {

        for (std::size_t i = 0; i < nx; ++i) {

            if (n == 3) {
                dg_wave_2D_bdry_kernel_element<3>(
                    &Up(i, 0), &Hp(i, 0), &Um(i, 0), &Hm(i, 0),
                    &dUpdt(i, 0), &dHpdt(i, 0), &dUmdt(i, 0), &dHmdt(i, 0),
                    &Cp(i, 0), &Cm(i, 0), Jx, w
                );
            } else if (n == 4) {
                dg_wave_2D_bdry_kernel_element<4>(
                    &Up(i, 0), &Hp(i, 0), &Um(i, 0), &Hm(i, 0),
                    &dUpdt(i, 0), &dHpdt(i, 0), &dUmdt(i, 0), &dHmdt(i, 0),
                    &Cp(i, 0), &Cm(i, 0), Jx, w
                );
            } else if (n == 5) {
                dg_wave_2D_bdry_kernel_element<5>(
                    &Up(i, 0), &Hp(i, 0), &Um(i, 0), &Hm(i, 0),
                    &dUpdt(i, 0), &dHpdt(i, 0), &dUmdt(i, 0), &dHmdt(i, 0),
                    &Cp(i, 0), &Cm(i, 0), Jx, w
                );
            } else if (n == 6) {
                dg_wave_2D_bdry_kernel_element<6>(
                    &Up(i, 0), &Hp(i, 0), &Um(i, 0), &Hm(i, 0),
                    &dUpdt(i, 0), &dHpdt(i, 0), &dUmdt(i, 0), &dHmdt(i, 0),
                    &Cp(i, 0), &Cm(i, 0), Jx, w
                );
            }

        }

    }
}


void  dg_wave_adjoint_2D_volume_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> u,
    py::array_t<double, py::array::c_style | py::array::forcecast> v,
    py::array_t<double, py::array::c_style | py::array::forcecast> h,
    py::array_t<double, py::array::c_style | py::array::forcecast> dudt,
    py::array_t<double, py::array::c_style | py::array::forcecast> dvdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> dhdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> c_arr,
    py::array_t<double, py::array::c_style | py::array::forcecast> D,
    double Jx,
    double Jy,
    double w
) {
    if (u.ndim() != 4)
        throw std::runtime_error("u must have shape (nx, ny, n, n)");

    const std::size_t nx = u.shape(0);
    const std::size_t ny = u.shape(1);
    const std::size_t n  = u.shape(2);

    if (u.shape(3) != n)
        throw std::runtime_error("Expected tensor shape (nx, ny, n, n)");

    if (D.ndim() != 2 || D.shape(0) != n || D.shape(1) != n)
        throw std::runtime_error("D must have shape (n, n)");

    // Fast accessors (no bounds checks)
    auto U = u.unchecked<4>();
    auto V = v.unchecked<4>();
    auto H = h.unchecked<4>();
    auto dUdt = dudt.mutable_unchecked<4>();
    auto dVdt = dvdt.mutable_unchecked<4>();
    auto dHdt = dhdt.mutable_unchecked<4>();
    auto C = c_arr.unchecked<4>();
    auto Dmat = D.unchecked<2>();

    // DG volume contraction
    #pragma omp parallel
    {

        #pragma omp for collapse(2)
        for (std::size_t i = 0; i < nx; ++i) {
            for (std::size_t j = 0; j < ny; ++j) {

                if (n == 3) {
                    dg_wave_adjoint_2D_volume_kernel_element<3>(
                        &U(i, j, 0, 0), &V(i, j, 0, 0), &H(i, j, 0, 0),
                        &dUdt(i, j, 0, 0), &dVdt(i, j, 0, 0), &dHdt(i, j, 0, 0),
                        &C(i, j, 0, 0), &Dmat(0, 0), Jx, Jy, w
                    );
                } else if (n == 4) {
                    dg_wave_adjoint_2D_volume_kernel_element<4>(
                        &U(i, j, 0, 0), &V(i, j, 0, 0), &H(i, j, 0, 0),
                        &dUdt(i, j, 0, 0), &dVdt(i, j, 0, 0), &dHdt(i, j, 0, 0),
                        &C(i, j, 0, 0), &Dmat(0, 0), Jx, Jy, w
                    );
                } else if (n == 5) {
                    dg_wave_adjoint_2D_volume_kernel_element<5>(
                        &U(i, j, 0, 0), &V(i, j, 0, 0), &H(i, j, 0, 0),
                        &dUdt(i, j, 0, 0), &dVdt(i, j, 0, 0), &dHdt(i, j, 0, 0),
                        &C(i, j, 0, 0), &Dmat(0, 0), Jx, Jy, w
                    );
                } else if (n == 6) {
                    dg_wave_adjoint_2D_volume_kernel_element<6>(
                        &U(i, j, 0, 0), &V(i, j, 0, 0), &H(i, j, 0, 0),
                        &dUdt(i, j, 0, 0), &dVdt(i, j, 0, 0), &dHdt(i, j, 0, 0),
                        &C(i, j, 0, 0), &Dmat(0, 0), Jx, Jy, w
                    );
                }

            }
        }
    }
}


void  dg_wave_adjoint_2D_bdry_kernel_py(
    py::array_t<double, py::array::c_style | py::array::forcecast> up,
    py::array_t<double, py::array::c_style | py::array::forcecast> hp,
    py::array_t<double, py::array::c_style | py::array::forcecast> um,
    py::array_t<double, py::array::c_style | py::array::forcecast> hm,
    py::array_t<double, py::array::c_style | py::array::forcecast> dupdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> dhpdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> dumdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> dhmdt,
    py::array_t<double, py::array::c_style | py::array::forcecast> cp,
    py::array_t<double, py::array::c_style | py::array::forcecast> cm,
    double Jx,
    double w
) {

    // Fast accessors (no bounds checks)
    auto Up = up.unchecked<2>();
    auto Hp = hp.unchecked<2>();
    auto Um = um.unchecked<2>();
    auto Hm = hm.unchecked<2>();

    auto dUpdt = dupdt.mutable_unchecked<2>();
    auto dHpdt = dhpdt.mutable_unchecked<2>();
    auto dUmdt = dumdt.mutable_unchecked<2>();
    auto dHmdt = dhmdt.mutable_unchecked<2>();

    auto Cp = cp.unchecked<2>();
    auto Cm = cm.unchecked<2>();

    const std::size_t nx = up.shape(0);
    const std::size_t n  = up.shape(1);

    // DG bdry contraction
    {

        for (std::size_t i = 0; i < nx; ++i) {

            if (n == 3) {
                dg_wave_adjoint_2D_bdry_kernel_element<3>(
                    &Up(i, 0), &Hp(i, 0), &Um(i, 0), &Hm(i, 0),
                    &dUpdt(i, 0), &dHpdt(i, 0), &dUmdt(i, 0), &dHmdt(i, 0),
                    &Cp(i, 0), &Cm(i, 0), Jx, w
                );
            } else if (n == 4) {
                dg_wave_adjoint_2D_bdry_kernel_element<4>(
                    &Up(i, 0), &Hp(i, 0), &Um(i, 0), &Hm(i, 0),
                    &dUpdt(i, 0), &dHpdt(i, 0), &dUmdt(i, 0), &dHmdt(i, 0),
                    &Cp(i, 0), &Cm(i, 0), Jx, w
                );
            } else if (n == 5) {
                dg_wave_adjoint_2D_bdry_kernel_element<5>(
                    &Up(i, 0), &Hp(i, 0), &Um(i, 0), &Hm(i, 0),
                    &dUpdt(i, 0), &dHpdt(i, 0), &dUmdt(i, 0), &dHmdt(i, 0),
                    &Cp(i, 0), &Cm(i, 0), Jx, w
                );
            } else if (n == 6) {
                dg_wave_adjoint_2D_bdry_kernel_element<6>(
                    &Up(i, 0), &Hp(i, 0), &Um(i, 0), &Hm(i, 0),
                    &dUpdt(i, 0), &dHpdt(i, 0), &dUmdt(i, 0), &dHmdt(i, 0),
                    &Cp(i, 0), &Cm(i, 0), Jx, w
                );
            }

        }

    }
}



PYBIND11_MODULE(_core, m) {
    m.doc() = "DG kernels";

    m.def("ader_dg_wave_2D_kernel",
        &ader_dg_wave_2D_kernel_py,
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
        py::arg("bdry_flag"),
        "Example DG tensor contraction in reference direction 0");

    m.def("ader_dg_wave_2D_kernel_adjoint",
        &ader_dg_wave_2D_kernel_adjoint_py,
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
        py::arg("bdry_flag"),
        "Example DG tensor contraction in reference direction 0");

    m.def("dg_wave_2D_volume_kernel",
        &dg_wave_2D_volume_kernel_py,
        py::arg("u"),
        py::arg("v"),
        py::arg("h"),
        py::arg("dudt"),
        py::arg("dvdt"),
        py::arg("dhdt"),
        py::arg("c"),
        py::arg("D"),
        py::arg("Jx"),
        py::arg("Jy"),
        py::arg("w"),
        "Example DG tensor contraction in reference direction 0");

    m.def("dg_wave_2D_bdry_kernel",
        &dg_wave_2D_bdry_kernel_py,
        py::arg("up"),
        py::arg("hp"),
        py::arg("um"),
        py::arg("hm"),
        py::arg("dupdt"),
        py::arg("dhpdt"),
        py::arg("dumdt"),
        py::arg("dhmdt"),
        py::arg("cp"),
        py::arg("cm"),
        py::arg("Jx"),
        py::arg("w"),
        "Example DG tensor contraction in reference direction 0");

    m.def("dg_wave_adjoint_2D_volume_kernel",
        &dg_wave_adjoint_2D_volume_kernel_py,
        py::arg("u"),
        py::arg("v"),
        py::arg("h"),
        py::arg("dudt"),
        py::arg("dvdt"),
        py::arg("dhdt"),
        py::arg("c"),
        py::arg("D"),
        py::arg("Jx"),
        py::arg("Jy"),
        py::arg("w"),
        "Example DG tensor contraction in reference direction 0");

    m.def("dg_wave_adjoint_2D_bdry_kernel",
        &dg_wave_adjoint_2D_bdry_kernel_py,
        py::arg("up"),
        py::arg("hp"),
        py::arg("um"),
        py::arg("hm"),
        py::arg("dupdt"),
        py::arg("dhpdt"),
        py::arg("dumdt"),
        py::arg("dhmdt"),
        py::arg("cp"),
        py::arg("cm"),
        py::arg("Jx"),
        py::arg("w"),
        "Example DG tensor contraction in reference direction 0");
}