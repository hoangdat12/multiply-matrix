#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

uint32_t reverse_bits(uint32_t x, int log2N) {
    uint32_t res = 0;
    for (int i = 0; i < log2N; i++) {
        res = (res << 1) | (x & 1);
        x >>= 1;
    }
    return res;
}

void bit_reverse_array(float complex* x, int N, int log2N) {
    for (int i = 0; i < N; i++) {
        uint32_t rev = reverse_bits(i, log2N);
        if (i < rev) {
            float complex temp = x[i];
            x[i] = x[rev];
            x[rev] = temp;
        }
    }
}

// W_N^k = e^(-j * 2*pi*k / N)
float complex* init_twiddle_factors(int N) {
    float complex* W = (float complex*)malloc((N / 2) * sizeof(float complex));
    if (W == NULL) {
        printf("Lỗi cấp phát bộ nhớ cho LUT!\n");
        exit(1);
    }
    
    for (int i = 0; i < N / 2; i++) {
        float angle = -2.0 * M_PI * i / N;
        W[i] = cosf(angle) + I * sinf(angle);
    }
    return W;
}

void fft_radix2_dit(float complex* x, float complex* W_lut, int N, int log2N) {
    bit_reverse_array(x, N, log2N);

    for (int s = 1; s <= log2N; s++) {
        int m = 1 << s;           // Số điểm của bài toán con ở stage hiện tại (2, 4, 8,...)
        int m_half = m >> 1;      // Một nửa kích thước (1, 2, 4,...)
        
        int twiddle_step = N / m; 

        for (int k = 0; k < N; k += m) {
            for (int j = 0; j < m_half; j++) {
                float complex w = W_lut[j * twiddle_step];
                
                int even_idx = k + j;
                int odd_idx  = k + j + m_half;
                
                float complex t = w * x[odd_idx];
                float complex u = x[even_idx];
                
                x[even_idx] = u + t;
                x[odd_idx]  = u - t;
            }
        }
    }
}

void print_complex_array(const char* label, float complex* arr, int N) {
    printf("%s\n", label);
    for (int i = 0; i < N; i++) {
        printf("X[%d] = %7.4f + %7.4fj\n", i, crealf(arr[i]), cimagf(arr[i]));
    }
    printf("\n");
}

int main() {
    int N = 8; // Số điểm FFT (phải là lũy thừa của 2)
    int log2N = (int)log2(N);
    
    float complex x[8] = {
        1.0 + 0.0*I,  1.0 + 0.0*I,  1.0 + 0.0*I,  1.0 + 0.0*I,
        0.0 + 0.0*I,  0.0 + 0.0*I,  0.0 + 0.0*I,  0.0 + 0.0*I
    };

    print_complex_array("Tín hiệu đầu vào x[n]:", x, N);
    float complex* W_lut = init_twiddle_factors(N);
    fft_radix2_dit(x, W_lut, N, log2N);
    print_complex_array("Kết quả FFT X[k]:", x, N);
    free(W_lut);

    return 0;
}