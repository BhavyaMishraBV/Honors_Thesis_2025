# fft_radix_2_3_5.py
# Copyright (c) 2025 Bhavya Mishra
# MIT License (see ../LICENSE). Advisor: Dr. Bart Horn.
#
# Mixed-radix (2, 3, 5) Cooley-Tukey FFT and inverse FFT for 1D and 3D arrays, written from scratch.
# Axis lengths must be products of 2, 3 and 5 (e.g. 25, 30, 64, 100). Non-cubic grids work.
#
# Both classes transform the array they are given IN PLACE (3D case) and expose the result as `.x`;
# pass a copy if you need to keep the input.
#
#   fwd = fft_radix(data.copy()).x          # forward FFT, same convention as numpy.fft.fftn
#   inv = inverseFFT_Radix(fwd.copy()).x    # inverse FFT, same convention as numpy.fft.ifftn
#
# Run `fft_radix_2_3_5.py` to check both transforms against NumPy.
import time
import numpy as np


# Forward transform (recursion tree in a class)
class fft_radix:
    def __init__(self, x):
        self.x = self.fft_3d(x)


    def fft_5(self,x):
        N = len(x)
        if N <= 1:
            return x.astype(complex)  # Ensure complex dtype
        elif N % 5 != 0:
            raise ValueError("Input size must be a multiple of 5 for radix-5 FFT")

        fifth = N // 5
        omega = np.exp(-2j * np.pi / N)
        omega_second = omega ** fifth
        omega_third = omega_second ** 2
        omega_fourth = omega_second ** 3
        omega_fifth = omega_second** 4


        # Split the input array into thirds
        x0 = x[0::5]
        x1 = x[1::5]
        x2 = x[2::5]
        x3 = x[3::5]
        x4 = x[4::5]

        # Perform FFTs on each third
        X0 = self.apply_radix_functions(x0)
        X1 = self.apply_radix_functions(x1)
        X2 = self.apply_radix_functions(x2)
        X3 = self.apply_radix_functions(x3)
        X4 = self.apply_radix_functions(x4)

        # Combine results
        result = np.zeros(N, dtype=complex)
        for k in range(fifth):
            w0 = omega ** k
            w1 = omega_second
            w2 = omega_third
            w3 = omega_fourth
            w4 = omega_fifth

            result[k] = (X0[k] + w0*X1[k] + w0*w0*X2[k]+ w0 * w0 * w0 * X3[k] + w0* w0 * w0 * w0 * X4[k])
            result[k + fifth] = (X0[k] + w0 * w1 * X1[k] + w0 * w0 * w2 * X2[k] + w0 * w0 * w0 * w3 * X3[k] + w0**4 * w4 * X4[k])
            result[k + 2 * fifth] = (X0[k] + w0 * w2 * X1[k] + w0 * w0 * w4 * X2[k] + w0 * w0 * w0 * w1 * X3[k] + w0**4 * w3* X4[k])
            result[k + 3 * fifth] = (X0[k] + w0 * w3 * X1[k] + w0 * w0 * w1 * X2[k] + w0 * w0 * w0 * w4 * X3[k] + w0**4 * w2* X4[k])
            result[k + 4 * fifth] = (X0[k] + w0 * w4 * X1[k] + w0 * w0 * w3 * X2[k] + w0 * w0 * w0 * w2 * X3[k] + w0**4 * w1* X4[k])
        return result
        #############################################################################

    def fft_3(self,x):
        N = len(x)
        if N <= 1:
            return x.astype(complex)  # Ensure complex dtype
        elif N % 3 != 0:
            raise ValueError("Input size must be a multiple of 3 for radix-3 FFT")

        third = N // 3
        omega = np.exp(-2j * np.pi / N)
        omega_third = omega ** third
        omega_second = omega_third ** 2

        # Split the input array into thirds
        x0 = x[0::3]
        x1 = x[1::3]
        x2 = x[2::3]

        # Perform FFTs on each third
        X0 = self.apply_radix_functions(x0)
        X1 = self.apply_radix_functions(x1)
        X2 = self.apply_radix_functions(x2)

        # Combine results
        result = np.zeros(N, dtype=complex)
        for k in range(third):
            w0 = omega ** k
            w1 = omega_third
            w2 = omega_second

            result[k] = (X0[k] + w0*X1[k] + w0*w0*X2[k])
            result[k + third] = (X0[k] + w0*w1 * X1[k] + w0*w0*w2 * X2[k])
            result[k + 2 * third] = (X0[k] + w0*w2 * X1[k] + w0*w0*w1 * X2[k])

        return result

    #############################################################################
    def fft_2(self,x):
        N = len(x)
        if N <= 1: return x
        even = self.apply_radix_functions(x[0::2])
        odd = self.apply_radix_functions(x[1::2])
        T = [np.exp(-2j * np.pi * k / N) * odd[k % (N // 2)] for k in range(N)]
        return [even[k % (N // 2)] + T[k] for k in range(N)]


    #############################################################################
    def prime_factors(self,n):
        factors = {}
        for p in [2, 3, 5]:
            while n % p == 0:
                if p in factors:
                    factors[p] += 1
                else:
                    factors[p] = 1
                n //= p
        if n > 1:
            raise ValueError("Input size must be a product of 2^n, 3^m, and/or 5^p.")
        return factors
        #############################################################################

    def apply_radix_functions(self,x):
        N = len(x)
        factors = self.prime_factors(N)

        if not factors:
            #raise ValueError("Input size must be a product of 2^n, 3^m, and/or 5^p.")
            return x

        # Apply radix FFTs
        for p, exp in factors.items():
            if p == 2:
                x = self.fft_2(x)
            elif p == 3:
                x = self.fft_3(x)
            elif p == 5:
                x = self.fft_5(x)
            break
        return x

    def fft_3d(self,x):

        if len(x.shape) == 1:
            # Apply 1D FFT to the input array
            return np.asarray(self.apply_radix_functions(x), dtype=complex)


        # Apply 1D FFT along each axis recursively
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j, :] = self.apply_radix_functions(x[i, j, :])
        for i in range(x.shape[0]):
            for k in range(x.shape[2]):
                x[i, :, k] = self.apply_radix_functions(x[i, :, k])
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x[:, j, k] = self.apply_radix_functions(x[:, j, k])

        return x


class inverseFFT_Radix:
    def __init__(self, x):
        self.x = self.inv_fft_3d(x)

    def radix_2_inv(self, x):
        N = len(x)
        if N <= 1: return x
        even = self.apply_inv_radix_functions(x[0::2])
        odd = self.apply_inv_radix_functions(x[1::2])
        T = [np.exp(2j * np.pi * k / N) * odd[k % (N // 2)] for k in range(N)]
        return [even[k % (N // 2)] + T[k] for k in range(N)]

    def radix_3_inv(self,x):
        N = len(x)
        if N <= 1:
            return x.astype(complex)  # Ensure complex dtype
        elif N % 3 != 0:
            raise ValueError("Input size must be a multiple of 3 for radix-3 FFT")

        third = N // 3
        omega = np.exp(2j * np.pi / N)
        omega_third = omega ** third
        omega_second = omega_third ** 2

        # Split the input array into thirds
        x0 = x[0::3]
        x1 = x[1::3]
        x2 = x[2::3]

        # Perform FFTs on each third
        X0 = self.apply_inv_radix_functions(x0)
        X1 = self.apply_inv_radix_functions(x1)
        X2 = self.apply_inv_radix_functions(x2)

        # Combine results
        result = np.zeros(N, dtype=complex)
        for k in range(third):
            w0 = omega ** k
            w1 = omega_third
            w2 = omega_second

            result[k] = (X0[k] + w0*X1[k] + w0*w0*X2[k])
            result[k + third] = (X0[k] + w0*w1 * X1[k] + w0*w0*w2 * X2[k])
            result[k + 2 * third] = (X0[k] + w0*w2 * X1[k] + w0*w0*w1 * X2[k])

        return result

    def radix_5_inv(self,x):
        N = len(x)
        if N <= 1:
            return x.astype(complex)  # Ensure complex dtype
        elif N % 5 != 0:
            raise ValueError("Input size must be a multiple of 5 for radix-5 FFT")

        fifth = N // 5
        omega = np.exp(2j * np.pi / N)
        omega_second = omega ** fifth
        omega_third = omega_second ** 2
        omega_fourth = omega_second ** 3
        omega_fifth = omega_second** 4


        # Split the input array into thirds
        x0 = x[0::5]
        x1 = x[1::5]
        x2 = x[2::5]
        x3 = x[3::5]
        x4 = x[4::5]

        # Perform FFTs on each third
        X0 = self.apply_inv_radix_functions(x0)
        X1 = self.apply_inv_radix_functions(x1)
        X2 = self.apply_inv_radix_functions(x2)
        X3 = self.apply_inv_radix_functions(x3)
        X4 = self.apply_inv_radix_functions(x4)

        # Combine results
        result = np.zeros(N, dtype=complex)
        for k in range(fifth):
            w0 = omega ** k
            w1 = omega_second
            w2 = omega_third
            w3 = omega_fourth
            w4 = omega_fifth

            result[k] = (X0[k] + w0*X1[k] + w0*w0*X2[k]+ w0 * w0 * w0 * X3[k] + w0* w0 * w0 * w0 * X4[k])
            result[k + fifth] = (X0[k] + w0 * w1 * X1[k] + w0 * w0 * w2 * X2[k] + w0 * w0 * w0 * w3 * X3[k] + w0**4 * w4 * X4[k])
            result[k + 2 * fifth] = (X0[k] + w0 * w2 * X1[k] + w0 * w0 * w4 * X2[k] + w0 * w0 * w0 * w1 * X3[k] + w0**4 * w3* X4[k])
            result[k + 3 * fifth] = (X0[k] + w0 * w3 * X1[k] + w0 * w0 * w1 * X2[k] + w0 * w0 * w0 * w4 * X3[k] + w0**4 * w2* X4[k])
            result[k + 4 * fifth] = (X0[k] + w0 * w4 * X1[k] + w0 * w0 * w3 * X2[k] + w0 * w0 * w0 * w2 * X3[k] + w0**4 * w1* X4[k])
        return result
        #############################################################################

    def prime_factors(self,n):
        factors = {}
        for p in [2, 3, 5]:
            while n % p == 0:
                if p in factors:
                    factors[p] += 1
                else:
                    factors[p] = 1
                n //= p
        if n > 1:
            raise ValueError("prime factor error: Input size must be a product of 2^n, 3^m, and/or 5^p.")
        return factors
        #############################################################################

    def apply_inv_radix_functions(self,x):
        N = len(x)
        factors = self.prime_factors(N)

        if not factors:
            #raise ValueError("Input size must be a product of 2^n, 3^m, and/or 5^p.")
            return x

        # Apply radix FFTs
        for p, exp in factors.items():
            if p == 2:
                x = self.radix_2_inv(x)
            elif p == 3:
                x = self.radix_3_inv(x)
            elif p == 5:
                x = self.radix_5_inv(x)
            break
        return x

    def inv_fft_3d(self, x):

        if len(x.shape) == 1:
            # 1D input: apply the inverse radix FFT directly
            return np.asarray(self.apply_inv_radix_functions(x), dtype=complex) / len(x)

        # Apply the inverse 1D FFT along each axis, dividing by that axis's own length
        # (so non-cubic grids are normalised correctly)
        n1, n2, n3 = x.shape
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                x[i, j, :] = self.apply_inv_radix_functions(x[i, j, :])
                x[i, j, :] = [y / n3 for y in x[i, j, :]]
        for i in range(x.shape[0]):
            for k in range(x.shape[2]):
                x[i, :, k] = self.apply_inv_radix_functions(x[i, :, k])
                x[i, :, k] = [y / n2 for y in x[i, :, k]]
        for j in range(x.shape[1]):
            for k in range(x.shape[2]):
                x[:, j, k] = self.apply_inv_radix_functions(x[:, j, k])
                x[:, j, k] = [y / n1 for y in x[:, j, k]]

        return x


############################################################################
def self_test():
    """Compare both transforms with NumPy on random data, and check forward -> inverse round trip."""
    rng = np.random.default_rng(0)
    ok = True
    print(f"{'shape':<14}{'forward':<10}{'inverse':<10}{'round trip':<12}")
    for shape in [(4, 4, 4), (5, 5, 5), (6, 6, 6), (9, 9, 9), (10, 10, 10),
                  (15, 15, 15), (25, 25, 25), (6, 10, 15), (12, 20, 5)]:
        x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        fwd = fft_radix(x.copy()).x
        good_f = np.allclose(fwd, np.fft.fftn(x))
        good_i = np.allclose(inverseFFT_Radix(x.copy()).x, np.fft.ifftn(x))
        good_r = np.allclose(inverseFFT_Radix(fwd.copy()).x, x)
        ok = ok and good_f and good_i and good_r
        print(f"{str(shape):<14}{'OK' if good_f else 'FAIL':<10}{'OK' if good_i else 'FAIL':<10}"
              f"{'OK' if good_r else 'FAIL':<12}")

    # 1D check
    for n in (20, 25, 30, 100):
        v = rng.standard_normal(n) + 0j
        good = (np.allclose(fft_radix(v.copy()).x, np.fft.fft(v)) and
                np.allclose(inverseFFT_Radix(v.copy()).x, np.fft.ifft(v)))
        ok = ok and good
        print(f"1D length {n:<5} {'OK' if good else 'FAIL'}")

    print("All tests passed." if ok else "Some tests FAILED.")
    return ok


def timing(n=25):
    """Time this FFT against NumPy on an n x n x n grid."""
    rng = np.random.default_rng(1)
    x = rng.standard_normal((n, n, n)) + 1j * rng.standard_normal((n, n, n))
    t0 = time.time()
    np.fft.fftn(x)
    t1 = time.time()
    fft_radix(x.copy())
    t2 = time.time()
    print(f"{n}^3 grid: NumPy {t1 - t0:.4f} s, this implementation {t2 - t1:.2f} s")


if __name__ == "__main__":
    self_test()
    # timing(25)   # uncomment to compare run times
