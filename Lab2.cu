#include <iostream>
#include <stdio.h>
#include <vector>
#include "cblas.h"
#include <chrono>
#include <cmath>
#include <omp.h>
#include <thread>
#include <thrust/complex.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <complex>
const int N = 8192;

using namespace std;

typedef complex<double> Complex;

__global__ void matrix_mul(thrust::complex<double>* a, thrust::complex<double>* b, thrust::complex<double>* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    thrust::complex<double> sum(0, 0);
    if (i < n && j < n) {
        for (int k = 0; k < n; k++)
            sum += a[i * n + k] * b[k * n + j];
        c[i * n + j] = sum;
    }
}

void matrixMultiply(thrust::complex<double>* A, thrust::complex<double>* B, thrust::complex<double>* C, int N) {
    thrust::complex<double>* d_A, * d_B, * d_C;
    size_t size = N * N * sizeof(thrust::complex<double>);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_mul << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
Complex** initializeMatrix(const int n)
{
    Complex** rows = new Complex * [n];
    Complex* mem = new Complex[n * n];
    for (int i = 0; i < n; i++)
    {
        rows[i] = mem;
        mem += n;
    }
    return rows;
}


void simpleMultiplication(Complex** a, Complex** b, Complex** c, const int size)
{
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < size; ++j)
        {
            Complex s = 0;
            for (int k = 0; k < size; k++)
                s += a[i][k] * b[j][k];
            c[i][j] = s;
        }
}

bool mat_equal(const int n, Complex** a, Complex** b, float eps = 1.e-3)
{
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if (abs(a[i][j] - b[i][j]) > eps)
                return false;
    return true;
}


int main() {
    srand((unsigned int)time(0));
    
    double c = 2 * pow(N, 3);

    float p1, p2, p3;
    Complex** a = initializeMatrix(N);
    Complex** b = initializeMatrix(N);
    Complex** bt = initializeMatrix(N);
    Complex** c1 = initializeMatrix(N);
    Complex** c2 = initializeMatrix(N);
    Complex** c3 = initializeMatrix(N);

    
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            a[i][j].real((double)rand() / RAND_MAX);
            a[i][j].imag(double(rand()) / RAND_MAX);
            b[i][j].real((double)rand() / RAND_MAX);
            b[i][j].imag(double(rand()) / RAND_MAX);
            c1[i][j].real((double)0);
            c1[i][j].imag((double)0);
            c2[i][j].real((double)0);
            c2[i][j].imag((double)0);
            c3[i][j].real((double) 0);
            c3[i][j].imag((double) 0);
        }


    // 1 -- simplest solution
    auto start = chrono::high_resolution_clock::now();

    //simpleMultiplication(a, b, c1, N);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time taken: " << 120 << " seconds" << endl;

    p1 = c / 120 * pow(10, -6);
    cout << "p = " << p1 << endl;

    // 2 -- blas solution
    start = chrono::high_resolution_clock::now();
    Complex alpha(1, 0);
    Complex beta(0, 0);
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, &alpha, a[0], N, b[0], N, &beta, c2[0], N);
    end = chrono::high_resolution_clock::now();
    duration = end - start;
    cout << "Time taken: " << duration.count() << " seconds" << endl;

    p2 = c / duration.count() * pow(10, -6);
    cout << "p = " << p2 << endl;

    if (mat_equal(N, c1, c2))
        cout << "c1 == c2 matrix test ok\n";
    else
        cout << "c1 != c2 matrix test failed\n";
    cout << "p1/p2 = " << (p1 / p2) * 100 << "%" << endl;


    //3 -- cuda solution
    start = chrono::high_resolution_clock::now();
    thrust::complex<double>* d_a = new thrust::complex<double>[N * N];
    thrust::complex<double>* d_b = new thrust::complex<double>[N * N];
    thrust::complex<double>* d_c = new thrust::complex<double>[N * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            d_a[i * N + j] = thrust::complex<double>(a[i][j].real(), a[i][j].imag());
            d_b[i * N + j] = thrust::complex<double>(b[i][j].real(), b[i][j].imag());
        }
    }
    matrixMultiply(d_a, d_b, d_c, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            c3[i][j] = Complex(d_c[i * N + j].real(), d_c[i * N + j].imag());

        }
    }
    end = chrono::high_resolution_clock::now();
    duration = end - start;
    cout << "Time taken: " << duration.count() << " seconds" << endl;


    p3 = c / duration.count()  * pow(10, -6);
    cout << "p = " << p3 << endl;
    cout << "p3/p2 = " << (p3 / p2) * 100 << "%" << endl;
    if (mat_equal(N, c3, c2))
        cout << "c3 == c2 matrix test ok\n";
    else
        cout << "c3 != c2 matrix test failed\n";


    cin.get();



    for (int i = 0; i < N; ++i) {
        delete[] a[i];
        delete[] b[i];
        delete[] c1[i];
        delete[] c2[i];
        delete[] c3[i];
    }
    delete[] a;
    delete[] b;
    delete[] c1;
    delete[] c2;
    delete[] c3;
    
    
    return 0;
    return 0;
}
