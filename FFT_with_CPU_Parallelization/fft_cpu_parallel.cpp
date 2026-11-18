// fft_cpu_parallel.cpp
// Copyright (c) 2025 Bhavya Mishra
// MIT License (see ../LICENSE). Advisor: Dr. Bart Horn.
//
// Multithreaded version of the mixed-radix (2, 3, 5) FFT and inverse FFT for 3D grids (see ../rd2 for the
// single-thread version). A 3D FFT is a set of independent 1D FFTs along each axis, so each axis pass
// distributes those 1D transforms across CPU threads with OpenMP. The recursive 1D routines use only
// local data, so they are safe to run concurrently.
//
// Each axis length must be a product of 2, 3 and 5 (e.g. 25, 30, 64, 100); grids need not be cubic.
//
//   fft_3d(grid)    forward transform, same convention as FFTW_FORWARD and numpy.fft.fftn (no normalisation)
//   ifft_3d(grid)   inverse transform, same convention as numpy.fft.ifftn (divided by n1*n2*n3)
//
// The program checks both transforms against FFTW3 and reports the speed-up from using several threads.
// Only the CPU version is implemented; the structure (independent slices) is what would map to a GPU.
//
// Build (Linux/macOS):  g++ -O3 -std=c++17 -fopenmp fft_cpu_parallel.cpp -o fft_parallel -lfftw3 -lm
// Build (Windows, MinGW):
//     g++ -O3 -std=c++17 -fopenmp fft_cpu_parallel.cpp -o fft_parallel.exe -I"<path-to-fftw>\include" "<path-to-fftw>\bin\libfftw3-3.dll"
//     (copy libfftw3-3.dll next to fft_parallel.exe)
// The OpenMP `collapse` clause needs OpenMP 3.0 or newer (g++ and clang; not the MSVC compiler).
//
// Usage:
//     ./fft_parallel                 run the full set of comparisons
//     ./fft_parallel N1 N2 N3       compare a single N1 x N2 x N3 grid
// Set the number of threads with the environment variable OMP_NUM_THREADS (default: all cores).

#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <chrono>
#include <algorithm>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <fftw3.h>

using namespace std;
using Complex = complex<double>;
const double PI = 3.141592653589793238460;

void fft_recursive(vector<Complex>& x, int sign);
void fft_radix2(vector<Complex>& x, int sign);
void fft_radix3(vector<Complex>& x, int sign);
void fft_radix5(vector<Complex>& x, int sign);
void fft_3d(vector<vector<vector<complex<double>>>>& x);    // forward
void ifft_3d(vector<vector<vector<complex<double>>>>& x);   // inverse (normalised)
bool validate_against_fftw(int n1, int n2, int n3, unsigned seed = 12345);
bool run_validation();

// A length is supported if it factors completely into 2, 3 and 5.
static bool supported_size(int n) {
    if (n < 1) return false;
    for (int p : {2, 3, 5}) while (n % p == 0) n /= p;
    return n == 1;
}

int main(int argc, char* argv[])
{
    if (argc == 4) {
        int n1 = atoi(argv[1]), n2 = atoi(argv[2]), n3 = atoi(argv[3]);
        if (!supported_size(n1) || !supported_size(n2) || !supported_size(n3)) {
            cerr << "Each size must be a product of 2, 3 and 5 (e.g. 25, 30, 64, 100)." << endl;
            return 2;
        }
        printf("Threads available: %d\n", omp_get_max_threads());
        return validate_against_fftw(n1, n2, n3) ? 0 : 1;
    }
    if (argc != 1) {
        cerr << "Usage: " << argv[0] << " [N1 N2 N3]" << endl;
        return 2;
    }
    printf("Threads available: %d\n", omp_get_max_threads());
    return run_validation() ? 0 : 1;
}

void fft_recursive(vector<Complex>& x, int sign) {
    int N = x.size();
    if (N <= 1) return;

    vector<int> factors;
    for (int p : {2, 3, 5}) {
        if (N % p == 0) {
            factors.push_back(p);
            while (N % p == 0) {
                N /= p;
            }
        }
    }

    if (factors.empty()) {
        return;
    }
    int p = factors[0];
    switch (p) {
    case 2:
        fft_radix2(x, sign);
        break;
    case 3:
        fft_radix3(x, sign);
        break;
    case 5:
        fft_radix5(x, sign);
        break;
    default:
        throw invalid_argument("Unsupported radix!");

    }
}
void fft_radix2(vector<Complex>& x, int sign) {
    int N = x.size();
    if (N <= 1) return;

    // Split the input into even and odd indices
    vector<Complex> even(N / 2), odd(N / 2);
    for (int i = 0; i < N / 2; i++) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    // Recursively perform FFT on the even and odd parts
    fft_recursive(even, sign);
    fft_recursive(odd, sign);

    // Combine results using the twiddle factors
    for (int k = 0; k < N / 2; ++k) {
        Complex t = polar(1.0, sign * 2 * PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}
void fft_radix3(vector<Complex>& x, int sign) {
    int N = x.size();
    if (N <= 1) return;

    int third = N / 3;
    vector<Complex> x0(third), x1(third), x2(third);
    for (int i = 0; i < third; i++) {
        x0[i] = x[i * 3];
        x1[i] = x[i * 3 + 1];
        x2[i] = x[i * 3 + 2];
    }

    // Recursive FFTs on each part
    fft_recursive(x0, sign);
    fft_recursive(x1, sign);
    fft_recursive(x2, sign);

    // Combine the results using twiddle factors
    Complex omega = exp(Complex(0, sign * 2 * PI / N));
    Complex omega_third = pow(omega, third);
    Complex omega_second = omega_third * omega_third;

    for (int k = 0; k < third; ++k) {
        Complex w0 = pow(omega, k);
        x[k] = x0[k] + w0 * x1[k] + w0 * w0 * x2[k];
        x[k + third] = x0[k] + w0 * omega_third * x1[k] + w0 * w0 * omega_second * x2[k];
        x[k + 2 * third] = x0[k] + w0 * omega_second * x1[k] + w0 * w0 * omega_third * x2[k];
    }
}
void fft_radix5(vector<Complex>& x, int sign) {
    int N = x.size();
    if (N <= 1) return;

    int fifth = N / 5;
    vector<Complex> x0(fifth), x1(fifth), x2(fifth), x3(fifth), x4(fifth);
    for (int i = 0; i < fifth; i++) {
        x0[i] = x[i * 5];
        x1[i] = x[i * 5 + 1];
        x2[i] = x[i * 5 + 2];
        x3[i] = x[i * 5 + 3];
        x4[i] = x[i * 5 + 4];
    }

    // Recursive FFTs on each part
    fft_recursive(x0, sign);
    fft_recursive(x1, sign);
    fft_recursive(x2, sign);
    fft_recursive(x3, sign);
    fft_recursive(x4, sign);

    // Combine the results using twiddle factors
    Complex wk = exp(Complex(0, sign * 2 * PI / N));
    Complex w1 = exp(Complex(0, sign * 2 * PI * fifth / N));
    Complex w2 = exp(Complex(0, sign * 4 * PI * fifth / N));
    Complex w3 = exp(Complex(0, sign * 6 * PI * fifth / N));
    Complex w4 = exp(Complex(0, sign * 8 * PI * fifth / N));

    for (int k = 0; k < fifth; ++k) {
        Complex w0 = pow(wk, k);
        x[k] = x0[k] + w0 * x1[k] + w0 * w0 * x2[k] + w0 * w0 * w0 * x3[k] + w0 * w0 * w0 * w0 * x4[k];
        x[k + fifth] = x0[k] + w0 * w1 * x1[k] + w0 * w0 * w2 * x2[k] + w0 * w0 * w0 * w3 * x3[k] + w0 * w0 * w0 * w0 * w4 * x4[k];
        x[k + 2 * fifth] = (x0[k] + w0 * w2 * x1[k] + w0 * w0 * w4 * x2[k] + w0 * w0 * w0 * w1 * x3[k] + w0 * w0 * w0 * w0 * w3 * x4[k]);
        x[k + 3 * fifth] = (x0[k] + w0 * w3 * x1[k] + w0 * w0 * w1 * x2[k] + w0 * w0 * w0 * w4 * x3[k] + w0 * w0 * w0 * w0 * w2 * x4[k]);
        x[k + 4 * fifth] = (x0[k] + w0 * w4 * x1[k] + w0 * w0 * w3 * x2[k] + w0 * w0 * w0 * w2 * x3[k] + w0 * w0 * w0 * w0 * w1 * x4[k]);
    }
}



// Applies the 1D transform along each axis of the grid. sign = -1: forward, sign = +1: inverse (unnormalised).
// The slices within one axis pass are independent, so the two outer loops are collapsed and shared among
// the threads. The three passes stay in order (each uses the output of the previous one); OpenMP waits for
// all threads at the end of each pass.
static void fft_3d_passes(vector<vector<vector<complex<double>>>>& x, int sign) {
    int dim1 = x.size();
    int dim2 = x[0].size();
    int dim3 = x[0][0].size();

    // Along axis 1 (i, j, :)
#pragma omp parallel for collapse(2)
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            vector<complex<double>> slice(dim3);
            for (int k = 0; k < dim3; ++k) {
                slice[k] = x[i][j][k];
            }
            fft_recursive(slice, sign);
            for (int k = 0; k < dim3; ++k) {
                x[i][j][k] = slice[k];
            }
        }
    }

    // Along axis 2 (i, :, k)
#pragma omp parallel for collapse(2)
    for (int i = 0; i < dim1; ++i) {
        for (int k = 0; k < dim3; ++k) {
            vector<complex<double>> slice2(dim2);
            for (int j = 0; j < dim2; ++j) {
                slice2[j] = x[i][j][k];
            }
            fft_recursive(slice2, sign);
            for (int j = 0; j < dim2; ++j) {
                x[i][j][k] = slice2[j];
            }
        }
    }

    // Along axis 3 (:, j, k)
#pragma omp parallel for collapse(2)
    for (int j = 0; j < dim2; ++j) {
        for (int k = 0; k < dim3; ++k) {
            vector<complex<double>> slice(dim1);
            for (int i = 0; i < dim1; ++i) {
                slice[i] = x[i][j][k];
            }
            fft_recursive(slice, sign);
            for (int i = 0; i < dim1; ++i) {
                x[i][j][k] = slice[i];
            }
        }
    }
}

// Forward 3D FFT, in place. Same convention as FFTW_FORWARD and numpy.fft.fftn (no normalisation).
void fft_3d(vector<vector<vector<complex<double>>>>& x) {
    fft_3d_passes(x, -1);
}

// Inverse 3D FFT, in place. Same convention as numpy.fft.ifftn: divided by n1*n2*n3, so ifft_3d(fft_3d(x)) = x.
void ifft_3d(vector<vector<vector<complex<double>>>>& x) {
    fft_3d_passes(x, +1);
    int dim1 = x.size(), dim2 = x[0].size(), dim3 = x[0][0].size();
    double scale = 1.0 / ((double)dim1 * dim2 * dim3);
#pragma omp parallel for collapse(2)
    for (int i = 0; i < dim1; ++i)
        for (int j = 0; j < dim2; ++j)
            for (int k = 0; k < dim3; ++k)
                x[i][j][k] *= scale;
}


// Same rule as numpy.allclose(a, b): |a - b| <= atol + rtol * |b| for every element
static bool all_close(const vector<Complex>& a, const vector<Complex>& b,
    double& max_abs_err, double rtol = 1e-5, double atol = 1e-8) {
    max_abs_err = 0.0;
    bool ok = true;
    for (size_t i = 0; i < a.size(); ++i) {
        double err = abs(a[i] - b[i]);
        max_abs_err = max(max_abs_err, err);
        if (err > atol + rtol * abs(b[i])) ok = false;
    }
    return ok;
}

static vector<Complex> flatten(const vector<vector<vector<Complex>>>& g) {
    vector<Complex> out;
    out.reserve(g.size() * g[0].size() * g[0][0].size());
    for (const auto& plane : g)
        for (const auto& row : plane)
            out.insert(out.end(), row.begin(), row.end());
    return out;
}

static vector<vector<vector<Complex>>> make_grid(const vector<Complex>& flat, int n1, int n2, int n3) {
    vector<vector<vector<Complex>>> g(n1, vector<vector<Complex>>(n2, vector<Complex>(n3)));
    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j)
            for (int k = 0; k < n3; ++k)
                g[i][j][k] = flat[((size_t)i * n2 + j) * n3 + k];
    return g;
}

static double seconds(chrono::high_resolution_clock::time_point a, chrono::high_resolution_clock::time_point b) {
    return chrono::duration<double>(b - a).count();
}

// Runs the forward and inverse FFT with 1 thread and with all available threads on the same random
// n1 x n2 x n3 grid, and FFTW3 once each way. Checks the forward result against FFTW_FORWARD, the inverse
// against FFTW_BACKWARD / (n1*n2*n3), that inverse(forward(x)) returns x, and that the multithreaded
// results equal the single-thread ones. Prints PASS/FAIL, the largest differences and the run times.
bool validate_against_fftw(int n1, int n2, int n3, unsigned seed) {
    mt19937 gen(seed);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    size_t total = (size_t)n1 * n2 * n3;
    int max_threads = omp_get_max_threads();

    vector<Complex> input(total);
    for (auto& v : input) v = Complex(dist(gen), dist(gen));

    auto fwd1 = make_grid(input, n1, n2, n3), invl1 = fwd1;
    auto fwdP = fwd1, invP = fwd1;

    // 1) Our FFT with one thread
    omp_set_num_threads(1);
    auto a0 = chrono::high_resolution_clock::now();
    fft_3d(fwd1);
    auto a1 = chrono::high_resolution_clock::now();
    ifft_3d(invl1);
    auto a2 = chrono::high_resolution_clock::now();

    // 2) Our FFT with all threads
    omp_set_num_threads(max_threads);
    auto b0 = chrono::high_resolution_clock::now();
    fft_3d(fwdP);
    auto b1 = chrono::high_resolution_clock::now();
    ifft_3d(invP);
    auto b2 = chrono::high_resolution_clock::now();
    auto round_trip = fwdP;
    ifft_3d(round_trip);

    // 3) FFTW3 on flat row-major arrays (same index order as the grid)
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * total);
    fftw_complex* outF = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * total);
    fftw_complex* outB = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * total);
    fftw_plan planF = fftw_plan_dft_3d(n1, n2, n3, in, outF, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan planB = fftw_plan_dft_3d(n1, n2, n3, in, outB, FFTW_BACKWARD, FFTW_ESTIMATE);
    for (size_t i = 0; i < total; ++i) { in[i][0] = input[i].real(); in[i][1] = input[i].imag(); }
    auto c0 = chrono::high_resolution_clock::now();
    fftw_execute(planF);
    auto c1 = chrono::high_resolution_clock::now();
    fftw_execute(planB);
    auto c2 = chrono::high_resolution_clock::now();

    // 4) Compare
    vector<Complex> refF(total), refB(total);
    for (size_t i = 0; i < total; ++i) {
        refF[i] = Complex(outF[i][0], outF[i][1]);
        refB[i] = Complex(outB[i][0], outB[i][1]) / (double)total;   // FFTW does not normalise
    }
    double errF, errB, errR, errT1, errT2;
    bool ok = all_close(flatten(fwdP), refF, errF);
    ok = all_close(flatten(invP), refB, errB, 1e-5, 1e-13) && ok;
    ok = all_close(flatten(round_trip), input, errR, 1e-5, 1e-13) && ok;
    ok = all_close(flatten(fwdP), flatten(fwd1), errT1, 1e-5, 1e-13) && ok;    // threads must not change the result
    ok = all_close(flatten(invP), flatten(invl1), errT2, 1e-5, 1e-13) && ok;

    double f1 = seconds(a0, a1), fP = seconds(b0, b1), i1 = seconds(a1, a2), iP = seconds(b1, b2);
    printf("%3d x %3d x %3d  %s  max|diff| fwd %.1e  inv %.1e  round trip %.1e\n",
        n1, n2, n3, ok ? "PASS" : "FAIL", errF, errB, errR);
    printf("      forward: 1 thread %.4f s, %d threads %.4f s (%.2fx)   inverse: 1 thread %.4f s, %d threads %.4f s (%.2fx)   FFTW: fwd %.4f s, inv %.4f s\n",
        f1, max_threads, fP, fP > 0 ? f1 / fP : 0.0, i1, max_threads, iP, iP > 0 ? i1 / iP : 0.0,
        seconds(c0, c1), seconds(c1, c2));

    fftw_destroy_plan(planF);
    fftw_destroy_plan(planB);
    fftw_free(in);
    fftw_free(outF);
    fftw_free(outB);
    return ok;
}

// Runs the checks for a range of sizes (powers of 2, 3 and 5, mixtures, and non-cubic grids).
// FFTW handles any size, so these are all valid comparisons.
bool run_validation() {
    int cubes[] = { 4, 5, 6, 8, 9, 10, 15, 25, 30, 50, 64, 100 };
    int boxes[][3] = { {6, 10, 15}, {12, 20, 5}, {16, 25, 9} };
    bool all_ok = true;
    for (int n : cubes)  all_ok = validate_against_fftw(n, n, n) && all_ok;
    for (auto& b : boxes) all_ok = validate_against_fftw(b[0], b[1], b[2]) && all_ok;
    printf(all_ok ? "\nAll comparisons with FFTW3 passed.\n" : "\nSome comparisons FAILED.\n");
    return all_ok;
}
