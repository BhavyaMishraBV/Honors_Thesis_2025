# Simulating the Universe: An Efficient Computational Model of the Pion Description of Large-Scale Structure

Honors thesis, Science Honors Program, Manhattan University (2025)
Author: Bhavya Mishra | Faculty advisor: Dr. Bart Horn

This repository holds my honors thesis and the C++ programs behind its FFT work. The thesis simulates the evolution of large-scale structure in the universe using the pion (velocity-potential) description, and focuses on making that simulation fast: moving from a Mathematica prototype to Python to C++, replacing O(N^6) Green's-function inversion with an FFT-based method (O(N^3 log N)), writing a custom mixed-radix FFT, and adding CPU parallelization.

- **Thesis:** [Thesis/Mishra_Honors_Thesis_2025.pdf](Thesis/Mishra_Honors_Thesis_2025.pdf)
- **Related paper:** Celik, Horn, Mishra & Muqattash (2026), [arXiv:2604.12009](https://arxiv.org/abs/2604.12009). Summary and links are in [Pion_research](https://github.com/BhavyaMishraBV/Pion_research).
- **Full simulation code (PLASTIC):** [bhorn9/pion_lss_code](https://github.com/bhorn9/pion_lss_code), written by Dr. Bart Horn and me. This repo does not duplicate it.

## Results documented in the thesis

- Moving from the Mathematica prototype to Python gave roughly a 20x speedup, and from Python to C++ roughly another 20x, about **400x overall**. A 25x25x25 grid took about 40 minutes in Python.
- A C++ test run from about 17 million to 1.63 billion years after the Big Bang took about 20 minutes.
- The custom radix-2/3/5 FFT removes the power-of-2 grid-size constraint and works directly on unflattened 3D arrays.
- Multithreading across independent FFT slices gave about a 3x gain on local machines.

## Checks of the code in this repository

The FFT programs include a built-in comparison against FFTW3 (forward and inverse transforms, plus a round-trip check) on grids from 4x4x4 to 100x100x100, including non-power-of-2 and non-cubic grids.

- The custom FFT agrees with FFTW3 to about 1e-12 on every test grid.
- FFTW3 is faster than the custom FFT on a single core (roughly 50x at 64^3 to 100^3 in my tests). The value of the custom version is its flexibility, not raw speed.
- With 8 threads, the multithreaded version ran about 2.6x to 3.3x faster than one thread on grids of 64^3 and larger.

Details and tables are in each folder's README.

## Repository contents

```
.
|-- README.md
|-- Thesis/
|   |-- Mishra_Honors_Thesis_2025.pdf
|   `-- Notice.txt
|-- Fast-12-Baseline/
|   |-- README.md
|   |-- LICENSE
|   `-- Source.cpp
|-- fft_colley2key_radix_implimentations/
|   |-- README.md
|   |-- LICENSE
|   |-- radix_2_3_5_integrated.cpp
|   `-- fft_radix_2_3_5.py
`-- FFT_with_CPU_Parallelization/
    |-- README.md
    |-- LICENSE
    `-- fft_cpu_parallel.cpp
```

| Folder | What it is | Authorship |
|---|---|---|
| `Thesis/` | The honors thesis (PDF) | Bhavya Mishra |
| `Fast-12-Baseline/` | Baseline simulation program that uses the FFTW3 library for its Fourier transforms. It is an early version of the code that later developed into PLASTIC. Start here to understand the structure and algorithm flow. | Co-authored by Dr. Bart Horn and Bhavya Mishra |
| `fft_colley2key_radix_implimentations/` | Mixed-radix (2, 3, 5) Cooley-Tukey FFT and inverse FFT on unflattened 3D grids, single CPU thread, in C++ and Python. | Bhavya Mishra (advisor: Dr. Bart Horn) |
| `FFT_with_CPU_Parallelization/` | Multithreaded (OpenMP) version of the FFT and inverse FFT, structured so it could later be extended to GPU platforms. A GPU version is not implemented. | Bhavya Mishra (advisor: Dr. Bart Horn) |

## Suggested order

1. Read `Fast-12-Baseline` to see the structure and algorithm flow.
2. Look at `fft_colley2key_radix_implimentations`, the custom FFT that can replace the FFTW3 calls: copy its `fft_3d` and `ifft_3d` functions (without the validation code and `main`) into the simulation.
3. Move on to `FFT_with_CPU_Parallelization` for the multithreaded version.

See each folder's README for its own build and usage notes. The programs build with g++, for example:

```bash
g++ -O3 -std=c++17 -fopenmp FFT_with_CPU_Parallelization/fft_cpu_parallel.cpp -o fft_parallel -lfftw3 -lm
```

## Requirements

- A C++17 compiler (g++), with OpenMP for the parallel version
- FFTW3, used by `Fast-12-Baseline` and by the validation code in the two FFT folders. FFTW3 is a separate GPL-licensed library and is not included in this repository.
- Python 3 with NumPy for the Python version
- A multi-core CPU to see the speed-up of the parallel version

## License and citation

The code in each program folder is released under the MIT license; each of those folders has its own `LICENSE` file, and the copyright holders are named at the top of each source file. `Fast-12-Baseline` is co-authored with Dr. Bart Horn. The thesis in `Thesis/` is (c) 2025 Bhavya Mishra, shared for reading, and is not covered by the MIT license; see `Thesis/Notice.txt`.

If you use this code, please cite the paper above.

## Credits

- **Bhavya Mishra:** author of the thesis; sole author of the two FFT programs; co-author of `Fast-12-Baseline`
- **Dr. Bart Horn:** faculty advisor; co-author of `Fast-12-Baseline`, of the PLASTIC simulation code and of the related paper
- Supported in part by the National Science Foundation's Research at Undergraduate Institutions (RUI) program under grant PHY-2210475, and by the Kakos School of Arts and Sciences Summer Research Scholars Program.
- Thanks to David Muqattash and Lara Celik for validating data for the results.
