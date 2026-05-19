#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>

// Example Cache L1 = 32KB = 32768 bytes
// Complex 8 bytes
// The total block is 3: A11 B11 C11
// Block size = 32768 / (3 * 8)
#define BLOCKSIZE 16 // bytes

typedef struct {
    int16_t real;
    int16_t imag;
} Complex;

// (a + jb)*(c + jd)
static Complex mul_complex(Complex a, Complex b) {
    Complex c;
    c.real = a.real * b.real - a.imag * b.imag;
    c.imag = a.real * b.imag + a.imag * b.real;
    return c;
}

// (a + jb) + (c + jd)
static Complex add_complex(Complex a, Complex b)
{
    Complex c;
    c.real = a.real + b.real;
    c.imag = a.imag + b.imag;
    return c;
}

static void mul_complex_avx2()


void multiply_complex(
    const Complex *A, const Complex *B, Complex *C,
    int M, int K, int N)
{
    for (int i = 0; i < M * N; i++) {
        C[i].real = 0.0f;
        C[i].imag = 0.0f;
    }

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                Complex t = mul_complex(A[i * K + k], B[k * N + j]);
                C[i * N + j] = add_complex(C[i * N + j], t);  // load + store mỗi lần
            }
        }
    }
}

void multiply_complex_V1(
    const Complex *A, const Complex *B, Complex *C,
    int M, int K, int N)
{
    for (int i = 0; i < M * N; i++) {
        C[i].real = 0.0f;
        C[i].imag = 0.0f;
    }

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            Complex temp = A[i * K + k];
            for (int j = 0; j < N; j++) {
                Complex t = mul_complex(temp, B[k * N + j]);
                C[i * N + j] = add_complex(C[i * N + j], t);
            }
        }
    }
}

void multiply_complex_V2(
    const Complex *A, const Complex *B, Complex *C,
    int M, int K, int N)
{
    for (int i = 0; i < M * N; i++) {
        C[i].real = 0.0f;
        C[i].imag = 0.0f;
    }

    // Slice
    for (int ii = 0; ii < M; ii += BLOCKSIZE) {
        int i_end = (ii + BLOCKSIZE < M) ? (ii + BLOCKSIZE) : M;

        for (int kk = 0; kk < K; kk += BLOCKSIZE) {
            int k_end = (kk + BLOCKSIZE < K) ? (kk + BLOCKSIZE) : K;

            for (int jj = 0; jj < N; jj += BLOCKSIZE) {
                int j_end = (jj + BLOCKSIZE < N) ? (jj + BLOCKSIZE) : N;

                // Multiply
                // Jump: rIdx * MAX_COL + cIdx
                for (int i = ii; i < i_end; i++) {
                    for (int k = kk; k < k_end; k++) {
                        Complex temp = A[i * K + k];

                        for (int j = jj; j < j_end; j++) {
                            Complex t = mul_complex(temp, B[k * N + j]);
                            C[i * N + j] = add_complex(C[i * N + j], t);
                        }
                    }
                }
            }
        }
    }
}

static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    srand(42);

    int M = 273 * 12, K = 4, N = 32;
    Complex *A = malloc(M * K * sizeof(Complex));
    Complex *B = malloc(K * N * sizeof(Complex));
    Complex *C = malloc(M * N * sizeof(Complex));
    Complex *D = malloc(M * N * sizeof(Complex));
    Complex *E = malloc(M * N * sizeof(Complex));

    if (!A || !B || !C || !D) {
        fprintf(stderr, "Allocation failed\n");
        free(A); free(B); free(C); free(D);
        return 1;
    }

    for (int i = 0; i < M * K; i++) {
        A[i].real = 1;
        A[i].imag = 2;
    }

    for (int i = 0; i < K * N; i++) {
        B[i].real = 3;
        B[i].imag = 4;
    }

    double t0 = get_time_sec();
    multiply_complex_V1(A, B, C, M, K, N);
    double t1 = get_time_sec();

    double t2 = get_time_sec();
    multiply_complex_V2(A, B, D, M, K, N);
    double t3 = get_time_sec();

    double t4 = get_time_sec();
    multiply_complex(A, B, E, M, K, N);
    double t5 = get_time_sec();

    printf("\nThoi gian chay:\n");
    printf("  V1: %.6f ms\n", (t1 - t0) * 1e3);
    printf("  V2: %.6f ms\n", (t3 - t2) * 1e3);
    printf("  Normal: %.6f ms\n", (t5 - t4) * 1e3);

    free(A); free(B); free(C); free(D); free(E);
    return 0;
}