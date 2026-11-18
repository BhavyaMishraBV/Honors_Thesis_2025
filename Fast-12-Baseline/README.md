# Fast-12

A compact, single-file C++ program that evolves the cosmic pion field on a periodic 3D grid, using the FFTW3 library for all Fourier transforms. It is an early version of the code that grew into **PLASTIC++** in [Dr. Bart Horn's repository](https://github.com/bhorn9/pion_lss_code/blob/main/PLASTIC%2B%2B.cpp), and it is the baseline for the custom FFTs in [`fft_colley2key_radix_implimentations`](fft_colley2key_radix_implimentations) and [`../FFT_with_CPU_Parallelization`](../FFT_with_CPU_Parallelization).

**Authors and copyright:** Dr. Bart Horn and Bhavya Mishra (co-authors and co-copyright holders). Released under the MIT license; see the header of `Source.cpp`.

## How it relates to PLASTIC++

Fast-12 follows the same algorithm as PLASTIC++ but keeps everything in one short file with the parameters written directly into `main()`, which makes it the easiest place to read how the method works. PLASTIC++ is the full version: it reads its settings from an `inputs.txt` file, adds Gaussian smoothing and an automatic stop, and writes separate output files for the pion field, its time derivative and the overdensity. For research use, run PLASTIC++. Use Fast-12 to learn the structure, and as the starting point for experimenting with the FFT replacements in the other folders.

## What it does

The pion field is the velocity potential of the cosmic matter fluid in a ΛCDM universe. At each time step Fast-12:

1. Computes gradient terms with finite differences and periodic boundary conditions.
2. Transforms to Fourier space with FFTW3, builds the Newtonian potential and overdensity, and solves for the nonlinear integral term with an inverse Laplacian applied in Fourier space.
3. Updates the pion field and its time derivative.
4. Removes high-wavenumber modes with a Fourier-space cutoff.
5. Stops when the scale factor reaches 1 (the present day), then writes the final pion field to `PidataC++.txt`.

## Input and output

- **Input:** a text file with the initial pion field on the grid, as Python-style nested lists (`[[[ ... ]]]`). Give its path as the first argument, or put a file named `ic_generated_random.txt` in the working directory. Initial-conditions generators such as `Gaussian_ICs.py` in the PLASTIC repository produce files like this; check that the format matches.
- **Output:** `PidataC++.txt`, with one grid row per line and values written as `real+imagi`.

## Parameters

Edit these at the top of `main()` in `Source.cpp`, then recompile:

| Parameter | Value |
|---|---|
| `boxsize` | 250 Mpc |
| `Omega_m` | 0.31 |
| `H0` | 68 km/s/Mpc |
| `z_init` | 99 |
| `meshpoints` | 64 (a 64 x 64 x 64 grid) |
| `cs_squared`, `cv_squared` | EFT parameters, currently 0 |

The grid-size benchmarks in the thesis used a 25 x 25 x 25 grid (`meshpoints = 25`).

## Build and run

Requires a C++17 compiler and FFTW3.

```bash
g++ -O3 -std=c++17 Source.cpp -o fast12 -lfftw3 -lm
./fast12 path/to/ic_generated_random.txt
```

On Ubuntu, install FFTW with `sudo apt install libfftw3-dev`. In Visual Studio, add the FFTW3 include and library paths to the project and link `libfftw3-3.lib`. Make sure only one source file with a `main()` is in the project.

## Related

- [`../rd2`](../rd2): standalone radix-2/3/5 FFT that can replace the FFTW3 calls
- [`../FFT_with_CPU_Parallelization`](../FFT_with_CPU_Parallelization): multithreaded FFT
- [PLASTIC++ and the full code](https://github.com/bhorn9/pion_lss_code)
- Paper: Celik, Horn, Mishra & Muqattash (2026), [arXiv:2604.12009](https://arxiv.org/abs/2604.12009)
