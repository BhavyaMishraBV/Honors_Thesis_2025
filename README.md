# Simulating the Universe: An Efficient Computational Model of the Pion Description of Large-Scale Structure

Honors thesis, Science Honors Program, Manhattan University (2025)
Author: Bhavya Mishra | Faculty advisor: Dr. Bart Horn

This repository holds my honors thesis and the C++ programs behind its FFT work. The thesis simulates the evolution of large-scale structure in the universe using the pion (velocity-potential) description, and focuses on making that simulation fast: moving from a Mathematica prototype to Python to C++, replacing O(N^6) Green's-function inversion with an FFT-based method (O(N^3 log N)), writing a custom mixed-radix FFT, and adding CPU parallelization.

- **Thesis:** [Mishra_Honors_Thesis_2025.pdf](Mishra_Honors_Thesis_2025.pdf)
- **Related paper:** Celik, Horn, Mishra & Muqattash (2026), [arXiv:2604.12009](https://arxiv.org/abs/2604.12009). Summary and links are in [Pion_research](https://github.com/BhavyaMishraBV/Pion_research).
- **Full simulation code (PLASTIC):** [bhorn9/pion_lss_code](https://github.com/bhorn9/pion_lss_code), written by Bart Horn and me. This repo does not duplicate it.

## Results documented in the thesis

- Moving from the Mathematica prototype to Python gave roughly a 20x speedup, and from Python to C++ roughly another 20x, about **400x overall**. A 25x25x25 grid took about 40 minutes in Python.
- A C++ test run from about 17 million to 1.63 billion years after the Big Bang took about 20 minutes.
- The custom radix-2/3/5 FFT has runtime comparable to NumPy and FFTW. Its purpose is flexibility: it removes the power-of-2 grid-size constraint and works directly on unflattened 3D arrays.
- Multithreading across independent FFT slices gave about a 3x gain on local machines.

## Repository contents

```
.
|-- README
|-- Mishra_Honors_Thesis_2025.pdf
|-- Fast-12/
|   |-- README
|   `-- Source.cpp
|-- rd2/
|   |-- README
|   `-- Source.cpp
`-- FFT_with_CPU_Parallelization/
    |-- README
    `-- Source.cpp
```

| Folder | What it is | Authorship |
|---|---|---|
| `Fast-12/` | Baseline simulation program that uses the FFTW3 library to transform pion-field data. It is an early version of the code that later developed into PLASTIC. Start here to understand the structure and algorithm flow. | Co-authored by Bart Horn and Bhavya Mishra |
| `rd2/` | Standalone FFT using radix-2, radix-3 and radix-5 (mixed-radix Cooley-Tukey) on a single CPU thread, supporting unflattened 3D grids. | Bhavya Mishra (advisor: Dr. Bart Horn) |
| `FFT_with_CPU_Parallelization/` | Multithreaded version of the FFT, structured so it could later be extended to GPU platforms. A GPU version is not implemented. | Bhavya Mishra (advisor: Dr. Bart Horn) |

## Suggested order

1. Read `Fast-12` to see the structure and algorithm flow.
2. Integrate `rd2`: include its source as a header in Fast-12 and replace the FFTW3 calls with rd2's forward and inverse transforms.
3. Move on to `FFT_with_CPU_Parallelization` for multithreading.

To compile the parallel version on Linux or macOS:

```bash
g++ -fopenmp -O3 -std=c++17 FFT_with_CPU_Parallelization/Source.cpp -o fft3d
```

On Windows (cmd):

```
g++ -fopenmp -O3 -std=c++17 FFT_with_CPU_Parallelization\Source.cpp -o fft3d.exe
```

See each folder's README for its own build and usage notes.

## Requirements

- A C++17 compiler (g++)
- FFTW3 (for `Fast-12` only)
- OpenMP support and the standard threading libraries (`<thread>`, `<mutex>`)
- A multi-core CPU for the parallel version

## License and citation

The code in this repository is released under the MIT license (see `LICENSE`). Copyright for each program is stated at the top of its `Source.cpp`: `Fast-12` is co-authored with Bart Horn, and `rd2` and `FFT_with_CPU_Parallelization` are my own work. The thesis PDF is (c) 2025 Bhavya Mishra, shared for reading, and is not covered by the MIT license.

If you use this code, please cite the paper above.

## Credits

- **Bhavya Mishra:** author of the thesis; sole author of `rd2` and `FFT_with_CPU_Parallelization`; co-author of `Fast-12`
- **Dr. Bart Horn:** faculty advisor; co-author of `Fast-12` and of the PLASTIC simulation code and the related paper
- Supported in part by the National Science Foundation's Research at Undergraduate Institutions (RUI) program under grant PHY-2210475, and by the Kakos School of Arts and Sciences Summer Research Scholars Program.
- Thanks to David Muqattash and Lara Celik for validating data for the results.
