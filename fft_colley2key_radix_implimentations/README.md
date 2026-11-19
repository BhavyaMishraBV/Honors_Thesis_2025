## mixed-radix (2, 3, 5) FFT for 3D grids using Cooley–Tukey FFT algorithm

A Cooley-Tukey FFT written from scratch that handles any grid whose side lengths are products of 2, 3 and 5 (for example 25, 30, 64 or 100), including non-cubic grids. It works directly on unflattened 3D arrays, so there is no power-of-2 size constraint and no need to flatten the data the way FFTW's plain interface does. It is the custom FFT from my honors thesis, built as a replacement for the FFTW3 calls in [Fast-12-Baseline](/Fast-12-Baseline).

**Author:** Bhavya Mishra (advisor: Dr. Bart Horn). MIT license, see the header of each file.

## Files

| File | What it is |
|---|---|
| `radix_2_3_5_integrated.cpp` | C++ forward FFT, single thread, with a built-in check against FFTW3 |
| `fft_radix_2_3_5` | Python version with both the forward and the inverse transform, with a self-test against NumPy |

The C++ version is forward-only. The inverse transform is in the Python file; a C++ inverse is planned.

## How it works

Each 1D transform splits the input by its smallest prime factor (2, 3 or 5), transforms the parts recursively, and combines them with twiddle factors. The 3D transform applies the 1D transform along each axis in turn. Conventions match `FFTW_FORWARD` and `numpy.fft.fftn` (no normalization on the forward transform).

## C++: build and validate

Requires a C++17 compiler and FFTW3.

Linux / macOS:

```bash
g++ -O3 -std=c++17 radix_2_3_5_integrated.cpp -o rd2 -lfftw3 -lm
./rd2
```

Windows (MinGW). Download the FFTW binaries from [fftw.org/install/windows.html](https://www.fftw.org/install/windows.html), then:

```
g++ -O3 -std=c++17 radix_2_3_5_integrated.cpp -o rd2.exe -I"<path-to-fftw>\include" "<path-to-fftw>\bin\libfftw3-3.dll"
copy "<path-to-fftw>\bin\libfftw3-3.dll" .
rd2.exe
```

`./rd2` compares the FFT with FFTW3 on random data for cubes from 4 to 100 and for three non-cubic grids, and prints PASS or FAIL, the largest difference and both run times. `./rd2 N1 N2 N3` checks a single grid, for example `./rd2 20 25 30`.

## Python: self-test

```bash
pip install numpy
python rd2_fft.py
```

This checks the forward transform, the inverse transform and the forward-then-inverse round trip against NumPy on several cubic and non-cubic grids, and checks 1D lengths too. Usage:

```python
fwd = fft_radix(data.copy()).x          # same as numpy.fft.fftn(data)
inv = inverseFFT_Radix(fwd.copy()).x    # same as numpy.fft.ifftn(fwd)
```

The classes transform the array they are given in place, so pass a copy if you need the original.

## Results

Accuracy: in the C++ comparison all 14 test grids (4^3 to 100^3, plus 6x10x15, 12x20x5 and 16x25x9) agree with FFTW3 to better than 1e-11 (largest difference 5e-12, at 100^3).

Speed (C++, `g++ -O3`, one thread, Windows laptop):

| Grid | This code | FFTW3 |
|---|---|---|
| 64 x 64 x 64 | 0.14 s | 0.0025 s |
| 100 x 100 x 100 | 0.71 s | 0.013 s |

FFTW3 is faster on a single core, by roughly 50x at these sizes: it is heavily tuned and vectorized, while this code is a straightforward recursive implementation. The value of this version is flexibility (any 2^a 3^b 5^c grid on unflattened 3D arrays) and a structure that is easy to parallelize; see [`../FFT_with_CPU_Parallelization`](../FFT_with_CPU_Parallelization). Timings depend on the machine and compiler flags, so run `./fft_colley2key_radix_implimentations` to get your own.
