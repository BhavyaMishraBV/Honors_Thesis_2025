// radix_2_3_5_integrated.cpp
// Copyright (c) 2025 Bhavya Mishra
// MIT License (see ../LICENSE). Advisor: Dr. Bart Horn.
//
// Mixed-radix (2, 3, 5) Cooley-Tukey forward FFT for 3D grids, written from scratch (single thread).
// Each axis length must be a product of 2, 3 and 5 (e.g. 25, 30, 64, 100); grids need not be cubic.
// The 3D transform is applied directly to a vector<vector<vector<complex<double>>>>, with no flattening.
// Same sign convention as FFTW_FORWARD and numpy.fft.fftn (no normalisation).
//
// The program checks the transform against FFTW3 on random data and prints the maximum difference
// and the run times of both methods.
//
// Build (Linux/macOS):  g++ -O3 -std=c++17 radix_2_3_5_integrated.cpp -o rd2 -lfftw3 -lm
// Build (Windows, MinGW):
//     g++ -O3 -std=c++17 radix_2_3_5_integrated.cpp -o rd2.exe -I"<path-to-fftw>\include" "<path-to-fftw>\bin\libfftw3-3.dll"
//     (copy libfftw3-3.dll next to rd2.exe)
//
// Usage:
//     ./rd2                 run the full set of comparisons with FFTW3
//     ./rd2 N1 N2 N3       compare a single N1 x N2 x N3 grid (each a product of 2, 3, 5)

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
#include <fftw3.h>

using namespace std;
using Complex = complex<double>;
const double PI = 3.141592653589793238460;

void fft_recursive(vector<Complex>& x);
void fft_radix2(vector<Complex>& x);
void fft_radix3(vector<Complex>& x);
void fft_radix5(vector<Complex>& x);
void fft_3d(vector<vector<vector<complex<double>>>>& x);
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
        return validate_against_fftw(n1, n2, n3) ? 0 : 1;
    }
    if (argc != 1) {
        cerr << "Usage: " << argv[0] << " [N1 N2 N3]" << endl;
        return 2;
    }
    return run_validation() ? 0 : 1;
}

void fft_recursive(vector<Complex>& x) {
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
        fft_radix2(x);
        break;
    case 3:
        fft_radix3(x);
        break;
    case 5:
        fft_radix5(x);
        break;
    default:
        throw invalid_argument("Unsupported radix!");

    }
}
void fft_radix2(vector<Complex>& x) {
    int N = x.size();
    if (N <= 1) return;

    // Split the input into even and odd indices
    vector<Complex> even(N / 2), odd(N / 2);
    for (int i = 0; i < N / 2; i++) {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    // Recursively perform FFT on the even and odd parts
    fft_recursive(even);
    fft_recursive(odd);

    // Combine results using the twiddle factors
    for (int k = 0; k < N / 2; ++k) {
        Complex t = polar(1.0, -2 * PI * k / N) * odd[k];
        x[k] = even[k] + t;
        x[k + N / 2] = even[k] - t;
    }
}
void fft_radix3(vector<Complex>& x) {
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
    fft_recursive(x0);
    fft_recursive(x1);
    fft_recursive(x2);

    // Combine the results using twiddle factors
    Complex omega = exp(Complex(0, -2 * PI / N));
    Complex omega_third = pow(omega, third);
    Complex omega_second = omega_third * omega_third;

    for (int k = 0; k < third; ++k) {
        Complex w0 = pow(omega, k);
        x[k] = x0[k] + w0 * x1[k] + w0 * w0 * x2[k];
        x[k + third] = x0[k] + w0 * omega_third * x1[k] + w0 * w0 * omega_second * x2[k];
        x[k + 2 * third] = x0[k] + w0 * omega_second * x1[k] + w0 * w0 * omega_third * x2[k];
    }
}
void fft_radix5(vector<Complex>& x) {
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
    fft_recursive(x0);
    fft_recursive(x1);
    fft_recursive(x2);
    fft_recursive(x3);
    fft_recursive(x4);

    // Combine the results using twiddle factors
    Complex wk = exp(Complex(0, -2 * PI / N));
    Complex w1 = exp(Complex(0, -2 * PI * fifth / N));
    Complex w2 = exp(Complex(0, -4 * PI * fifth / N));
    Complex w3 = exp(Complex(0, -6 * PI * fifth / N));
    Complex w4 = exp(Complex(0, -8 * PI * fifth / N));

    for (int k = 0; k < fifth; ++k) {
        Complex w0 = pow(wk, k);
        x[k] = x0[k] + w0 * x1[k] + w0 * w0 * x2[k] + w0 * w0 * w0 * x3[k] + w0 * w0 * w0 * w0 * x4[k];
        x[k + fifth] = x0[k] + w0 * w1 * x1[k] + w0 * w0 * w2 * x2[k] + w0 * w0 * w0 * w3 * x3[k] + w0 * w0 * w0 * w0 * w4 * x4[k];
        x[k + 2 * fifth] = (x0[k] + w0 * w2 * x1[k] + w0 * w0 * w4 * x2[k] + w0 * w0 * w0 * w1 * x3[k] + w0 * w0 * w0 * w0 * w3 * x4[k]);
        x[k + 3 * fifth] = (x0[k] + w0 * w3 * x1[k] + w0 * w0 * w1 * x2[k] + w0 * w0 * w0 * w4 * x3[k] + w0 * w0 * w0 * w0 * w2 * x4[k]);
        x[k + 4 * fifth] = (x0[k] + w0 * w4 * x1[k] + w0 * w0 * w3 * x2[k] + w0 * w0 * w0 * w2 * x3[k] + w0 * w0 * w0 * w0 * w1 * x4[k]);
    }
}


void fft_3d(vector<vector<vector<complex<double>>>>& x) {
    int dim1 = x.size();
    int dim2 = x[0].size();
    int dim3 = x[0][0].size();

    // Along axis 1 (i, j, :)
    for (int i = 0; i < dim1; ++i) {
        for (int j = 0; j < dim2; ++j) {
            vector<complex<double>> slice(dim3);
            for (int k = 0; k < dim3; ++k) {
                slice[k] = x[i][j][k];
            }
            fft_recursive(slice);
            for (int k = 0; k < dim3; ++k) {
                x[i][j][k] = slice[k];
            }
        }
    }

    // Along axis 2 (i, :, k)
    for (int i = 0; i < dim1; ++i) {
        for (int k = 0; k < dim3; ++k) {
            vector<complex<double>> slice2(dim2);
            for (int j = 0; j < dim2; ++j) {
                slice2[j] = x[i][j][k];
            }
            fft_recursive(slice2);
            for (int j = 0; j < dim2; ++j) {
                x[i][j][k] = slice2[j];
            }
        }
    }

    // Along axis 3 (:, j, k)
    for (int j = 0; j < dim2; ++j) {
        for (int k = 0; k < dim3; ++k) {
            vector<complex<double>> slice(dim1);
            for (int i = 0; i < dim1; ++i) {
                slice[i] = x[i][j][k];
            }
            fft_recursive(slice);
            for (int i = 0; i < dim1; ++i) {
                x[i][j][k] = slice[i];
            }
        }
    }
}
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

// Runs the mixed-radix fft_3d and FFTW3's forward 3D FFT on the same random n1 x n2 x n3 grid.
// Prints PASS/FAIL, the largest absolute difference, and both run times. Returns true on PASS.
bool validate_against_fftw(int n1, int n2, int n3, unsigned seed) {
    mt19937 gen(seed);
    uniform_real_distribution<double> dist(-1.0, 1.0);
    size_t total = (size_t)n1 * n2 * n3;

    // Same random input for both methods
    vector<Complex> input(total);
    for (auto& v : input) v = Complex(dist(gen), dist(gen));

    // 1) Our FFT on a vector<vector<vector<>>> grid
    vector<vector<vector<Complex>>> grid(n1, vector<vector<Complex>>(n2, vector<Complex>(n3)));
    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j)
            for (int k = 0; k < n3; ++k)
                grid[i][j][k] = input[((size_t)i * n2 + j) * n3 + k];

    auto t0 = chrono::high_resolution_clock::now();
    fft_3d(grid);
    auto t1 = chrono::high_resolution_clock::now();

    // 2) FFTW3 on a flat row-major array (same index order as the grid)
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * total);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * total);
    fftw_plan plan = fftw_plan_dft_3d(n1, n2, n3, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    for (size_t i = 0; i < total; ++i) { in[i][0] = input[i].real(); in[i][1] = input[i].imag(); }

    auto t2 = chrono::high_resolution_clock::now();
    fftw_execute(plan);
    auto t3 = chrono::high_resolution_clock::now();

    // 3) Compare
    vector<Complex> mine(total), ref(total);
    for (int i = 0; i < n1; ++i)
        for (int j = 0; j < n2; ++j)
            for (int k = 0; k < n3; ++k)
                mine[((size_t)i * n2 + j) * n3 + k] = grid[i][j][k];
    for (size_t i = 0; i < total; ++i) ref[i] = Complex(out[i][0], out[i][1]);

    double max_err;
    bool ok = all_close(mine, ref, max_err);

    printf("%3d x %3d x %3d   %s   max |diff| = %.3e   ours %.4f s   FFTW %.4f s\n",
        n1, n2, n3, ok ? "PASS" : "FAIL", max_err,
        chrono::duration<double>(t1 - t0).count(),
        chrono::duration<double>(t3 - t2).count());

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
    return ok;
}

// Runs the checks for a range of sizes (powers of 2, 3 and 5, mixtures, and non-cubic grids).
// FFTW handles any size, so these are all valid comparisons.
bool run_validation() {
    int cubes[] = { 4, 5, 6, 8, 9, 10, 15, 25, 30, 64, 100 };
    int boxes[][3] = { {6, 10, 15}, {12, 20, 5}, {16, 25, 9} };
    bool all_ok = true;
    for (int n : cubes)  all_ok = validate_against_fftw(n, n, n) && all_ok;
    for (auto& b : boxes) all_ok = validate_against_fftw(b[0], b[1], b[2]) && all_ok;
    printf(all_ok ? "\nAll comparisons with FFTW3 passed.\n" : "\nSome comparisons FAILED.\n");
    return all_ok;
}