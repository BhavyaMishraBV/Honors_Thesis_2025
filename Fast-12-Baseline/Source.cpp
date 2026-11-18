// Fast-12/Source.cpp
// Copyright (c) 2025 Bart Horn and Bhavya Mishra
// MIT License (see ../LICENSE). Early version of the pion-field simulation code that developed into PLASTIC.
// Uses FFTW3 for the 3D transforms.

#include <iostream>
#include <cmath>
#include <vector>
#include <fftw3.h>
#include <chrono>
#include <complex>
#include<fstream>
#include <string>
#include <sstream>
#include<algorithm>


using namespace std;
const double pi = 3.14159265358979323846;



int indexWrap(int index, int size) {
    // Wrap negative indices to the valid range
    return (index % size + size) % size;
}




std::string removeChars(const std::string& str, const std::string& charsToRemove) {
    std::string result;
    for (char c : str) {
        if (charsToRemove.find(c) == std::string::npos) {
            result += c;
        }
    }
    return result;
}

// Function to split a string into a vector of strings based on a delimiter
std::vector<std::string> split(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(str);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::vector<std::vector<std::complex<double>>>> create3DVector(int meshpoints) {
    return std::vector<std::vector<std::vector<std::complex<double>>>>(meshpoints,
        std::vector<std::vector<std::complex<double>>>(meshpoints,
            std::vector<std::complex<double>>(meshpoints, std::complex<double>(0.0, 0.0))));
}
//vector<vector<vector<complex<double>>>> create3DVector(int size) 
//{
//    return vector<vector<vector<complex<double>>>>(size, vector<vector<complex<double>>>(size, vector<complex<double>>(size)));
//}
vector<complex<double>> create1DVector(int size)
{
    return vector<complex<double>>(size);
}
vector<double> create1DVectorReal(int size)
{
    return vector<double>(size);
}


// Flatten function for 3D vector of doubles to 1D vector of complex doubles
std::vector<std::complex<double>> flatten(int meshpoints, const std::vector<std::vector<std::vector<double>>>& integrand) {
    std::vector<std::complex<double>> integrand_flat(meshpoints * meshpoints * meshpoints);
    for (int i = 0; i < meshpoints; i++) {
        for (int j = 0; j < meshpoints; j++) {
            for (int k = 0; k < meshpoints; k++) {
                int index = i * meshpoints * meshpoints + j * meshpoints + k;
                integrand_flat[index] = std::complex<double>(integrand[i][j][k], 0.0);
            }
        }
    }
    return integrand_flat;
}

// Flatten function for 3D vector of complex doubles to 1D vector of complex doubles
std::vector<std::complex<double>> flatten(int meshpoints, const std::vector<std::vector<std::vector<std::complex<double>>>>& integrand) {
    std::vector<std::complex<double>> integrand_flat(meshpoints * meshpoints * meshpoints);
    for (int i = 0; i < meshpoints; i++) {
        for (int j = 0; j < meshpoints; j++) {
            for (int k = 0; k < meshpoints; k++) {
                int index = i * meshpoints * meshpoints + j * meshpoints + k;
                integrand_flat[index] = integrand[i][j][k];
            }
        }
    }
    return integrand_flat;
}

// Unflatten function to convert 1D vector of complex doubles back to 3D vector of complex doubles
std::vector<std::vector<std::vector<std::complex<double>>>> unflatten(int meshpoints, const std::vector<std::complex<double>>& integrand_flat) {
    std::vector<std::vector<std::vector<std::complex<double>>>> integrand(meshpoints, std::vector<std::vector<std::complex<double>>>(meshpoints, std::vector<std::complex<double>>(meshpoints)));
    for (int i = 0; i < meshpoints; ++i) {
        for (int j = 0; j < meshpoints; ++j) {
            for (int k = 0; k < meshpoints; ++k) {
                int index = i * meshpoints * meshpoints + j * meshpoints + k;
                integrand[i][j][k] = integrand_flat[index];
            }
        }
    }
    return integrand;
}
void print_complex_array(fftw_complex* arr, int size) {
    for (int i = 0; i < size; ++i) {
        std::cout << "(" << arr[i][0] << ", " << arr[i][1] << ") ";
    }
    std::cout << std::endl;

}

int main(int argc, char* argv[])
{
    cout << "let's start\n\n";
    double boxsize = 250.0; // Mpc
    double Omega_m = 0.31;
    double H0 = 68.0; // km/s/Mpc
    double z_init = 99.0; // should be less than 1000
    const int meshpoints = 64;
    double cutoff = 8.52; // Leave this fixed for now
    double cs_squared = 0E-6;
    double cv_squared = 0E-6;
    int ic_from_output = 1;
    int total_points = meshpoints * meshpoints * meshpoints;

    double meshsize = boxsize / meshpoints;
    cout << "Size of periodic box is: " << boxsize << " Mpc" << endl;
    std::cout << "Number of grid points is: " << meshpoints << std::endl;

    double a_init = 1.0 / (1.0 + z_init);
    double Hubble_init = (H0 / 299792.458) * a_init * std::sqrt(Omega_m / std::pow(a_init, 3));
    double time_init = 2.0 * std::sqrt(a_init) * 299792.458 / (H0 * std::sqrt(Omega_m));
    double time = time_init;

    double mesh[meshpoints][meshpoints][meshpoints][3] = { 0.0 };
    std::vector<std::vector<std::vector<double>>> ic1(
        meshpoints, std::vector<std::vector<double>>(
            meshpoints, std::vector<double>(meshpoints, 0.0)
        )
    );

    std::vector<std::vector<std::vector<double>>> icprime1(
        meshpoints, std::vector<std::vector<double>>(
            meshpoints, std::vector<double>(meshpoints, 0.0)
        )
    );

    // Path to the initial-conditions text file: first command-line argument, or a file in the working directory.
    std::string filePath = (argc > 1) ? argv[1] : "ic_generated_random.txt";
    std::ifstream file(filePath);
    if (!file) {
        std::cerr << "Unable to open file " << filePath << std::endl;
        return 1;
    }
    else {
        std::cout << "file opened" << std::endl;
    }

    std::string fileContent((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();

    // Parse the file content assuming Python-style [[[...]]] formatting
    std::vector<std::string> arr;
    size_t pos = 0;
    while ((pos = fileContent.find("[[[", pos)) != std::string::npos) {
        size_t start = pos + 3;
        size_t end = fileContent.find("]]]", start);
        if (end == std::string::npos) break;
        arr.push_back(fileContent.substr(start, end - start));
        pos = end + 3;
    }

    if (!arr.empty()) {
        std::stringstream ss(arr[0]);
        std::string layer;

        int i = 0;
        while (std::getline(ss, layer, ']') && i < meshpoints) {
            if (layer.find('[') == std::string::npos) continue;

            std::replace(layer.begin(), layer.end(), '[', ' ');
            std::stringstream layerStream(layer);
            std::string row;
            int j = 0;

            while (std::getline(layerStream, row, ',') && j < meshpoints) {
                std::stringstream rowStream(row);
                std::string numStr;
                int k = 0;

                while (std::getline(rowStream, numStr, ' ') && k < meshpoints) {
                    try {
                        if (!numStr.empty()) {
                            ic1[i][j][k] = std::stod(numStr);
                            ++k;
                        }
                    }
                    catch (...) {}
                }
                ++j;
            }
            ++i;
        }
    }

    // Compute icprime = ic / time_init
    for (int i = 0; i < meshpoints; ++i) {
        for (int j = 0; j < meshpoints; ++j) {
            for (int k = 0; k < meshpoints; ++k) {
                icprime1[i][j][k] = ic1[i][j][k] / time_init;
            }
        }
    }


    auto ic = flatten(meshpoints, ic1);
    auto icprime = flatten(meshpoints, icprime1);
    auto pidata = create1DVector(meshpoints * meshpoints * meshpoints);
    auto pidatanext = create1DVector(total_points);

    vector<complex<double>> piprimedata(total_points, 0.0);
    pidata = ic;
    pidatanext = pidata;
    piprimedata = icprime;
    auto piprimedatanext = icprime;

    time = time_init;
    double a_scale = a_init;

    int numsteps = 6000;

    auto overdensity = create1DVector(total_points);
    auto theta = create1DVector(total_points);
    auto delta = create1DVector(total_points);
    vector<complex<double>> NewtonPhi(total_points, 0.0);
    vector<complex<double>> pi_FT(total_points, 0.0);
    vector<complex<double>> delta_FT(total_points, 0.0);
    vector<complex<double>> theta_FT(total_points, 0.0);
    vector<complex<double>> pi_prime_FT(total_points, 0.0);
    vector<complex<double>> integrand(total_points, 0.0);
    vector<complex<double>> FTinverseLaplacian(total_points, 0.0);
    vector<complex<double>> integral(total_points, 0.0);
    vector<complex<double>> gradsquared(total_points, 0.0);

    auto start = chrono::high_resolution_clock::now();
    auto in = create1DVector(meshpoints * meshpoints * meshpoints);
    fftw_plan plan_forward_NewtonPhi = fftw_plan_dft_3d(meshpoints, meshpoints, meshpoints, reinterpret_cast<fftw_complex*>(ic.data()), reinterpret_cast<fftw_complex*>(in.data()), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_forward_NewtonPhi);
    fftw_destroy_plan(plan_forward_NewtonPhi);
    for (int t = 1; t < numsteps; t++)
    {

        if (a_scale > 1.0) {
            break;
        }
        double a_step = a_scale * 0.01;
        double timestep = (a_step * 299792.458) / ((H0)*pow(a_scale, 2) * sqrt((Omega_m / pow(a_scale, 3)) + (1 - Omega_m)));
        double Hubble = (H0 / 299792.458) * a_scale * sqrt((Omega_m / pow(a_scale, 3)) + (1 - Omega_m));
        double Hubble_prime = (1 - ((3 / 2) * (Omega_m / (Omega_m + (1 - Omega_m) * pow(a_scale, 3))))) * pow(Hubble, 2);

        int term = 0;
        for (int i = 0; i < meshpoints; i++) {
            for (int j = 0; j < meshpoints; j++) {
                for (int k = 0; k < meshpoints; k++) {
                    // Flattened index for current point
                    int idx = i * meshpoints * meshpoints + j * meshpoints + k;

                    // Calculate indices for neighboring points with periodic boundary conditions
                    int idx_ip1 = ((i + 1) % meshpoints) * meshpoints * meshpoints + j * meshpoints + k; // i+1
                    int idx_im1 = ((i - 1 + meshpoints) % meshpoints) * meshpoints * meshpoints + j * meshpoints + k; // i-1
                    int idx_jp1 = i * meshpoints * meshpoints + ((j + 1) % meshpoints) * meshpoints + k; // j+1
                    int idx_jm1 = i * meshpoints * meshpoints + ((j - 1 + meshpoints) % meshpoints) * meshpoints + k; // j-1
                    int idx_kp1 = i * meshpoints * meshpoints + j * meshpoints + ((k + 1) % meshpoints); // k+1
                    int idx_km1 = i * meshpoints * meshpoints + j * meshpoints + ((k - 1 + meshpoints) % meshpoints); // k-1

                    // Compute gradient squared using neighbors in 1D flattened arrays
                    gradsquared[idx] += (1.0 / (4.0 * pow(meshsize, 2))) * (
                        pow(abs(pidata[idx_ip1] - pidata[idx_im1]), 2) +
                        pow(abs(pidata[idx_jp1] - pidata[idx_jm1]), 2) +
                        pow(abs(pidata[idx_kp1] - pidata[idx_km1]), 2)
                        );

                    NewtonPhi[idx] += -(piprimedata[idx] + (Hubble)*pidata[idx] + (1.0 / 2.0) * gradsquared[idx]);
                }
            }
        }

        fftw_plan plan_forward_NewtonPhi = fftw_plan_dft_3d(meshpoints, meshpoints, meshpoints, reinterpret_cast<fftw_complex*>(NewtonPhi.data()), reinterpret_cast<fftw_complex*>(NewtonPhi.data()), FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan_forward_NewtonPhi);

        fftw_plan plan_forward_pi = fftw_plan_dft_3d(meshpoints, meshpoints, meshpoints, reinterpret_cast<fftw_complex*>(pidata.data()), reinterpret_cast<fftw_complex*>(pi_FT.data()), FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan_forward_pi);

        double factor = 6.28318 * 6.28318 / (boxsize * boxsize);
        for (int i = 0; i < meshpoints; i++) {
            for (int j = 0; j < meshpoints; j++) {
                for (int k = 0; k < meshpoints; k++) {
                    int idx = i * meshpoints * meshpoints + j * meshpoints + k;
                    int idx_i = std::min(i, meshpoints - i);
                    int idx_j = std::min(j, meshpoints - j);
                    int idx_k = std::min(k, meshpoints - k);
                    double k_squared = idx_i * idx_i + idx_j * idx_j + idx_k * idx_k;

                    delta_FT[idx] = (-NewtonPhi[idx] * factor * k_squared) / ((Hubble * Hubble - Hubble_prime) * boxsize * boxsize);
                    theta_FT[idx] = (-pi_FT[idx] * factor * k_squared) / (boxsize * boxsize);
                }
            }

        }

        fftw_plan plan_backward_delta = fftw_plan_dft_3d(meshpoints, meshpoints, meshpoints, reinterpret_cast<fftw_complex*>(delta_FT.data()), reinterpret_cast<fftw_complex*>(overdensity.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan_backward_delta);

        fftw_plan plan_backward_theta = fftw_plan_dft_3d(meshpoints, meshpoints, meshpoints, reinterpret_cast<fftw_complex*>(theta_FT.data()), reinterpret_cast<fftw_complex*>(theta.data()), FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(plan_backward_theta);

        fftw_destroy_plan(plan_forward_NewtonPhi);
        fftw_destroy_plan(plan_forward_pi);
        fftw_destroy_plan(plan_backward_delta);
        fftw_destroy_plan(plan_backward_theta);
        fftw_cleanup();

        double norm_factor = 1.0 / (meshpoints * meshpoints * meshpoints);
        for (int i = 0; i < meshpoints; i++) {
            for (int j = 0; j < meshpoints; j++) {
                for (int k = 0; k < meshpoints; k++) {
                    int idx = i * meshpoints * meshpoints + j * meshpoints + k;
                    overdensity[idx] = delta_FT[idx].real() * norm_factor;
                    if (overdensity[idx].real() < -1.0) {
                        cout << "num of steps are:: " << numsteps - 6000;
                        cout << "Underdensity is less than -1.0 at time" << time;
                    }
                    delta[idx] = ((Hubble_prime - Hubble * Hubble)) * overdensity[idx];
                    theta[idx] = theta_FT[idx].real() * norm_factor;
                }
            }
        }
        //compute integrand
        for (int i = 0; i < meshpoints; i++) {
            for (int j = 0; j < meshpoints; j++) {
                for (int k = 0; k < meshpoints; k++) {
                    // Compute the 1D index for (i, j, k)
                    int idx = i * meshpoints * meshpoints + j * meshpoints + k;

                    // Apply periodic boundary conditions using indexWrap
                    int i_wrap = indexWrap(i, meshpoints);
                    int j_wrap = indexWrap(j, meshpoints);
                    int k_wrap = indexWrap(k, meshpoints);

                    int idx_wrap = i_wrap * meshpoints * meshpoints + j_wrap * meshpoints + k_wrap;

                    // Add theta * delta term
                    integrand[idx] += theta[idx] * delta[idx];

                    // Compute the indices for the neighboring points
                    int idx_ip2 = indexWrap(i + 2, meshpoints) * meshpoints * meshpoints + j_wrap * meshpoints + k_wrap;
                    int idx_im2 = indexWrap(i - 2, meshpoints) * meshpoints * meshpoints + j_wrap * meshpoints + k_wrap;

                    int idx_jp2 = i_wrap * meshpoints * meshpoints + indexWrap(j + 2, meshpoints) * meshpoints + k_wrap;
                    int idx_jm2 = i_wrap * meshpoints * meshpoints + indexWrap(j - 2, meshpoints) * meshpoints + k_wrap;

                    int idx_kp2 = i_wrap * meshpoints * meshpoints + j_wrap * meshpoints + indexWrap(k + 2, meshpoints);
                    int idx_km2 = i_wrap * meshpoints * meshpoints + j_wrap * meshpoints + indexWrap(k - 2, meshpoints);

                    int idx_ip1 = indexWrap(i + 1, meshpoints) * meshpoints * meshpoints + j_wrap * meshpoints + k_wrap;
                    int idx_im1 = indexWrap(i - 1, meshpoints) * meshpoints * meshpoints + j_wrap * meshpoints + k_wrap;

                    int idx_jp1 = i_wrap * meshpoints * meshpoints + indexWrap(j + 1, meshpoints) * meshpoints + k_wrap;
                    int idx_jm1 = i_wrap * meshpoints * meshpoints + indexWrap(j - 1, meshpoints) * meshpoints + k_wrap;

                    int idx_kp1 = i_wrap * meshpoints * meshpoints + j_wrap * meshpoints + indexWrap(k + 1, meshpoints);
                    int idx_km1 = i_wrap * meshpoints * meshpoints + j_wrap * meshpoints + indexWrap(k - 1, meshpoints);

                    integrand[idx_wrap] += (1.0 / (4.0 * meshsize * meshsize)) * (
                        (pidata[idx_ip2] + pidata[idx_im2] +
                            pidata[idx_jp2] + pidata[idx_jm2] +
                            pidata[idx_kp2] + pidata[idx_km2] -
                            6.0 * pidata[idx_wrap]) * delta[idx_wrap] +

                        (pidata[idx_ip1] - pidata[idx_im1]) * (delta[idx_ip1] - delta[idx_im1]) +
                        (pidata[idx_jp1] - pidata[idx_jm1]) * (delta[idx_jp1] - delta[idx_jm1]) +
                        (pidata[idx_kp1] - pidata[idx_km1]) * (delta[idx_kp1] - delta[idx_km1])
                        );
                }
            }
        }

        fftw_plan forwardPlan = fftw_plan_dft_3d(meshpoints, meshpoints, meshpoints,
            reinterpret_cast<fftw_complex*>(integrand.data()),
            reinterpret_cast<fftw_complex*>(integrand.data()),
            FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(forwardPlan);
        fftw_destroy_plan(forwardPlan);


        for (int i = 0; i < meshpoints; i++) {
            for (int j = 0; j < meshpoints; j++) {
                for (int k = 0; k < meshpoints; k++) {
                    int index = i * meshpoints * meshpoints + j * meshpoints + k;
                    FTinverseLaplacian[index] = (-integrand[index] * pow(boxsize, 2)) /
                        (pow(6.28318, 2) * (pow(std::min(i, meshpoints - i), 2) +
                            pow(std::min(j, meshpoints - j), 2) +
                            pow(std::min(k, meshpoints - k), 2) +
                            pow(0.001, 2)));
                }
            }
        }
        // Create FFTW plan for inverse transform
        fftw_plan inversePlan = fftw_plan_dft_3d(meshpoints, meshpoints, meshpoints,
            reinterpret_cast<fftw_complex*>(FTinverseLaplacian.data()),
            reinterpret_cast<fftw_complex*>(FTinverseLaplacian.data()),
            FFTW_BACKWARD, FFTW_ESTIMATE);
        fftw_execute(inversePlan);
        fftw_destroy_plan(inversePlan);

        for (int i = 0; i < meshpoints; i++) {
            for (int j = 0; j < meshpoints; j++) {
                for (int k = 0; k < meshpoints; ++k) {
                    int index = i * meshpoints * meshpoints + j * meshpoints + k;
                    integral[index] = real(FTinverseLaplacian[index]) / (meshpoints * meshpoints * meshpoints);
                }
            }
        }
        // The 'integral' vector now contains the result of the inverse FFT

        for (int i = 0; i < meshpoints; i++) {
            for (int j = 0; j < meshpoints; j++) {
                for (int k = 0; k < meshpoints; k++) {
                    int idx = i * meshpoints * meshpoints + j * meshpoints + k;
                    // Compute the wrapped 1D indices for neighbors
                    int idx_ip1 = indexWrap(i + 1, meshpoints) * meshpoints * meshpoints + j * meshpoints + k;
                    int idx_im1 = indexWrap(i - 1, meshpoints) * meshpoints * meshpoints + j * meshpoints + k;

                    int idx_jp1 = i * meshpoints * meshpoints + indexWrap(j + 1, meshpoints) * meshpoints + k;
                    int idx_jm1 = i * meshpoints * meshpoints + indexWrap(j - 1, meshpoints) * meshpoints + k;

                    int idx_kp1 = i * meshpoints * meshpoints + j * meshpoints + indexWrap(k + 1, meshpoints);
                    int idx_km1 = i * meshpoints * meshpoints + j * meshpoints + indexWrap(k - 1, meshpoints);

                    int idx_ip2 = indexWrap(i + 2, meshpoints) * meshpoints * meshpoints + j * meshpoints + k;
                    int idx_im2 = indexWrap(i - 2, meshpoints) * meshpoints * meshpoints + j * meshpoints + k;

                    int idx_jp2 = i * meshpoints * meshpoints + indexWrap(j + 2, meshpoints) * meshpoints + k;
                    int idx_jm2 = i * meshpoints * meshpoints + indexWrap(j - 2, meshpoints) * meshpoints + k;

                    int idx_kp2 = i * meshpoints * meshpoints + j * meshpoints + indexWrap(k + 2, meshpoints);
                    int idx_km2 = i * meshpoints * meshpoints + j * meshpoints + indexWrap(k - 2, meshpoints);

                    pidatanext[idx] += piprimedata[idx] * timestep;
                    piprimedatanext[idx] += (-(2 * Hubble) * piprimedata[idx] - (2 * Hubble_prime + 0 * pow(Hubble, 2)) * pidata[idx]) * timestep;
                    piprimedatanext[idx] += -(Hubble / 2) * (gradsquared[idx]) * (timestep);

                    // Gradient squared terms
                    piprimedatanext[idx] += -(1.0 / (4.0 * meshsize * meshsize)) * (
                        (pidata[idx_ip1] - pidata[idx_im1]) * (piprimedata[idx_ip1] - piprimedata[idx_im1]) +
                        (pidata[idx_jp1] - pidata[idx_jm1]) * (piprimedata[idx_jp1] - piprimedata[idx_jm1]) +
                        (pidata[idx_kp1] - pidata[idx_km1]) * (piprimedata[idx_kp1] - piprimedata[idx_km1])) * timestep;

                    // Integral term
                    piprimedatanext[idx] += -(integral[idx]) * timestep;

                    // Second gradient term using wrapped indices
                    piprimedatanext[idx] += ((5.0 / 3.0) * cs_squared + (1.0 / 3.0) * cv_squared) *
                        (1.0 / (4.0 * meshsize * meshsize)) * (
                            pidata[idx_ip2] + pidata[idx_im2] +
                            pidata[idx_jp2] + pidata[idx_jm2] +
                            pidata[idx_kp2] + pidata[idx_km2] - 6.0 * pidata[idx]);

                    // Third gradient term using wrapped indices
                    piprimedatanext[idx] += ((1.0 / 3.0) * cs_squared + (1.0 / 2.0) * cv_squared) * time *
                        (1.0 / (4.0 * meshsize * meshsize)) * (
                            piprimedata[idx_ip2] + piprimedata[idx_im2] +
                            piprimedata[idx_jp2] + piprimedata[idx_jm2] +
                            piprimedata[idx_kp2] + piprimedata[idx_km2] - 6.0 * piprimedata[idx]);
                }
            }
        }

        fftw_plan plan_forward_update_pi = fftw_plan_dft_3d(
            meshpoints,
            meshpoints,
            meshpoints,
            reinterpret_cast<fftw_complex*>(pidatanext.data()),
            reinterpret_cast<fftw_complex*>(pi_FT.data()),
            FFTW_FORWARD,
            FFTW_ESTIMATE
        );
        fftw_plan plan_forward_update_pi_prime = fftw_plan_dft_3d(
            meshpoints,
            meshpoints,
            meshpoints,
            reinterpret_cast<fftw_complex*>(piprimedatanext.data()),
            reinterpret_cast<fftw_complex*>(pi_prime_FT.data()),
            FFTW_FORWARD,
            FFTW_ESTIMATE
        );

        fftw_execute(plan_forward_update_pi);
        fftw_execute(plan_forward_update_pi_prime);

        for (int i = 0; i < meshpoints; ++i) {
            for (int j = 0; j < meshpoints; ++j) {
                for (int k = 0; k < meshpoints; ++k) {
                    double k_mag = (6.28318 / boxsize) * std::sqrt(std::pow(std::min(i, meshpoints - i), 2) +
                        std::pow(std::min(j, meshpoints - j), 2) +
                        std::pow(std::min(k, meshpoints - k), 2) + 1E-24);
                    if (k_mag > (6.28318 / cutoff)) {
                        int idx = i * meshpoints * meshpoints + j * meshpoints + k;
                        pi_FT[idx] = std::complex<double>(0.0, 0.0);
                        pi_prime_FT[idx] = std::complex<double>(0.0, 0.0);
                    }
                }
            }
        }
        fftw_plan plan_inverse_pi = fftw_plan_dft_3d(
            meshpoints,
            meshpoints,
            meshpoints,
            reinterpret_cast<fftw_complex*>(pi_FT.data()),
            reinterpret_cast<fftw_complex*>(pi_FT.data()),
            FFTW_BACKWARD,
            FFTW_ESTIMATE
        );
        fftw_execute(plan_inverse_pi);

        // Normalize and extract the real part of the result into pidatanext
        int N = meshpoints * meshpoints * meshpoints;
        for (int i = 0; i < meshpoints; ++i) {
            for (int j = 0; j < meshpoints; ++j) {
                for (int k = 0; k < meshpoints; ++k) {
                    int idx = i * meshpoints * meshpoints + j * meshpoints + k;
                    pidatanext[idx] = pi_FT[idx].real() / N;
                }
            }
        }

        fftw_plan plan_inverse_pi_prime = fftw_plan_dft_3d(
            meshpoints,
            meshpoints,
            meshpoints,
            reinterpret_cast<fftw_complex*>(pi_prime_FT.data()),
            reinterpret_cast<fftw_complex*>(pi_prime_FT.data()),
            FFTW_BACKWARD,
            FFTW_ESTIMATE
        );

        // Execute the inverse FFT plan
        fftw_execute(plan_inverse_pi_prime);
        // Normalize and extract the real part of the result into pidatanext
        for (int i = 0; i < meshpoints; ++i) {
            for (int j = 0; j < meshpoints; ++j) {
                for (int k = 0; k < meshpoints; ++k) {
                    int idx = i * meshpoints * meshpoints + j * meshpoints + k;
                    piprimedatanext[idx] = pi_prime_FT[idx].real() / N;
                }
            }
        }


        // Clean up the FFTW plan
        fftw_destroy_plan(plan_inverse_pi);
        fftw_destroy_plan(plan_inverse_pi_prime);
        fftw_destroy_plan(plan_forward_update_pi);
        fftw_destroy_plan(plan_forward_update_pi_prime);

        time = time + timestep;
        a_scale += a_step;
        numsteps += 1;
        auto pi = pidatanext;
        pidata = pi;
        auto piprime = piprimedatanext;
        piprimedata = piprime;

    }



    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << endl << endl << "time is :::: " << duration.count() << endl;
    cout << "num of steps are:: " << numsteps - 6000;

    ofstream outputFile("PidataC++.txt");

    //Check if the file was opened successfully
    if (!outputFile) {
        cerr << "Error opening the file." << endl;
        return 1;
    }
    auto pidata2 = unflatten(meshpoints, pidata);
    for (int i = 0; i < meshpoints; i++) {
        for (int j = 0; j < meshpoints; j++) {
            for (int k = 0; k < meshpoints; k++) {
                outputFile << pidata2[i][j][k].real() << "+" << pidata2[i][j][k].imag() << "i ";
            }
            outputFile << "\n";
        }
    }

    // Close the file
    outputFile.close();


    return 0;
}

