## FFT_with_CPU_Parallelization: multithreaded mixed-radix FFT

Here I discuss the mixed-radix (2, 3, 5) FFT and inverse FFT from [radix_2_3_5_integrated](fft_colley2key_radix_implimentatins/radix_2_3_5_integrated.cpp), with the work spread across CPU threads using OpenMP. A 3D FFT is a set of independent 1D FFTs along each axis, so each axis pass hands those 1D transforms to different threads. It works on any grid whose side lengths are products of 2, 3 and 5 (for example 25, 30, 64 or 100), including non-cubic grids.

**Author:** Bhavya Mishra (advisor: Dr. Bart Horn). MIT license, see the header of `fft_cpu_parallel.cpp`.

## How it works

- The 1D routines (`fft_recursive`, `fft_radix2/3/5`) are identical to the single-thread version in `rd2` and use only local data, so any number of threads can run them at the same time.
- `fft_3d` makes three passes, along axis 1, axis 2 and axis 3. Within a pass, the independent slices are divided among the threads (`#pragma omp parallel for collapse(2)`). The passes themselves stay in order because each one uses the output of the previous pass.
- The structure (many independent slices) is what would map onto a GPU. A GPU version is **not** implemented; this folder is the CPU version only.
- `fft_3d(grid)` is the forward transform (same convention as `FFTW_FORWARD` and `numpy.fft.fftn`, no normalization). `ifft_3d(grid)` is the inverse (same convention as `numpy.fft.ifftn`, divided by the number of grid points). The inverse uses the same routines with the opposite sign in the twiddle factors. Both are parallelized.

## Build and run

Requires a C++17 compiler with OpenMP 3.0 or newer (g++ or clang) and FFTW3.

Linux / macOS:

```bash
g++ -O3 -std=c++17 -fopenmp fft_cpu_parallel.cpp -o fft_parallel -lfftw3 -lm
./fft_parallel
```

Windows (MinGW). Download the FFTW binaries from [fftw.org/install/windows.html](https://www.fftw.org/install/windows.html), then:

```
g++ -O3 -std=c++17 -fopenmp fft_cpu_parallel.cpp -o fft_parallel.exe -I"<path-to-fftw>\include" "<path-to-fftw>\bin\libfftw3-3.dll"
copy "<path-to-fftw>\bin\libfftw3-3.dll" .
fft_parallel.exe
```

On macOS the system compiler has no OpenMP; use `brew install libomp` and add `-Xpreprocessor -fopenmp -lomp`, or install GCC.

`./fft_parallel` runs a set of comparisons on random data (cubes from 4 to 100 and three non-cubic grids). For each grid it checks the forward transform against `FFTW_FORWARD`, the inverse against `FFTW_BACKWARD` divided by the grid size, and that the inverse of the forward transform returns the input. It prints PASS or FAIL, the largest differences, the time with one thread and with all threads for both transforms, the speed-up, and the FFTW3 times. A PASS also requires the multithreaded results to match the single-thread results.

`./fft_parallel N1 N2 N3` checks a single grid, for example `./fft_parallel 50 50 50`. Control the thread count with the environment variable `OMP_NUM_THREADS`.

## Results

Accuracy: on all 15 test grids (4^3 to 100^3, plus 6x10x15, 12x20x5 and 16x25x9) the multithreaded forward and inverse transforms agree with FFTW3 (largest forward difference 4.8e-12, at 100^3; largest inverse difference 1.3e-16), the inverse of the forward transform recovers the input (largest error 4.9e-15), and the multithreaded results match the single-thread results.

Speed (`g++ -O3 -fopenmp`, 8 threads available, Windows machine, one run per size):

| Grid | Forward, 1 thread | Forward, 8 threads | Speed-up | Inverse speed-up | FFTW3 forward |
|---|---|---|---|---|---|
| 25 x 25 x 25 | 0.0076 s | 0.0045 s | 1.7x | 1.7x | 0.0002 s |
| 50 x 50 x 50 | 0.061 s | 0.032 s | 1.9x | 2.6x | 0.0022 s |
| 64 x 64 x 64 | 0.15 s | 0.059 s | 2.6x | 2.9x | 0.0031 s |
| 100 x 100 x 100 | 0.56 s | 0.17 s | 3.3x | 3.0x | 0.019 s |

- Multithreading gives about a 3x speed-up on grids of 64^3 and larger, in line with the roughly 3x reported in the thesis.
- Very small grids (about 6^3 and below) run slower with threads, because starting the threads costs more than the work.
- FFTW3 is still faster than this code: about 9x at 100^3 with 8 threads here, and about 30x against one thread. The multithreading narrows that gap but does not close it.
- Timings depend on the machine, the compiler and the number of cores, and vary from run to run, so run `./fft_parallel` to get your own.

## Related

- [fft radix method](fft_colley2key_radix_implimentation): the single-thread version and the Python version
- [Baseline algorithm](Fast-12-Baseline): the simulation program that uses FFTW3, which these FFTs are meant to replace allowing us to scale the algorithm develoved in PLASCTIC.cpp
