#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>
#include <stdbool.h>

/**
 * Đoạn code optimize nhân 2 ma trận theo chuẩn 3GPP
 * Kích thước Resource Grid: 3276 x 4
 * Kích thước W = 4 x 32
 * Kích thước output = 3276 x 32
 * 
 * Optimization:
 *   - Loop i→k→j    : A[i][k] giữ trong register, B và C duyệt tuần tự
 *                     K=4 nhỏ → toàn bộ W (256B) nằm trong L1 cache
 *   - Block Tiling   : Chia nhỏ ma trận để tận dụng và lưu dữ liệu trong Cache
 *   - AVX2 Intrinsics: Thay vì sử dụng các phép tuần tự của C, Intrinsics hỗ trợ
 *                     gọi SIMD instructions qua các function được định nghĩa sẵn
 *                     trong thư viện immintrin để thực hiện song song các phép toán
 *   - Build -o3 -o1 o2
*/

#define BLOCK_SIZE 32
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

// __attribute__((optimize("no-tree-vectorize")))
void multiply_ikj_block(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real, int16_t *C_imag,
    int M, int K, int N)
{
    for (int i = 0; i < M * N; i++) C_real[i] = C_imag[i] = 0;

    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        int i_end = MIN(ii + BLOCK_SIZE, M);
        for (int kk = 0; kk < K; kk += BLOCK_SIZE) {
            int k_end = MIN(kk + BLOCK_SIZE, K);
            for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
                int j_end = MIN(jj + BLOCK_SIZE, N);

                for (int i = ii; i < i_end; i++)
                    for (int k = kk; k < k_end; k++)
                        for (int j = jj; j < j_end; j++) {
                            C_real[i*N+j] += A_real[i*K+k] * B_real[k*N+j] - A_imag[i*K+k] * B_imag[k*N+j];
                            C_imag[i*N+j] += A_real[i*K+k] * B_imag[k*N+j] + A_imag[i*K+k] * B_real[k*N+j];
                        }
            }
        }
    }
}

void multiply_ikj_avx_block(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real, int16_t *C_imag,
    int M, int K, int N)
{
  
    for (int i = 0; i < M * N; i++) C_real[i] = C_imag[i] = 0;

    const int AVX_READ = 16;

    for (int ii = 0; ii < M; ii += BLOCK_SIZE) {
        int i_end = MIN(ii + BLOCK_SIZE, M);

        for (int jj = 0; jj < N; jj += BLOCK_SIZE) {
            int j_end = MIN(jj + BLOCK_SIZE, N);

            // C[i][j] += A[i][k] * B[k][j]
            for (int i = ii; i < i_end; i++) {
                for (int k = 0; k < K; k++) {
                    __m256i ar = _mm256_set1_epi16(A_real[i*K+k]);
                    __m256i ai = _mm256_set1_epi16(A_imag[i*K+k]);
                    
                    int j = jj;
                    for (; j <= j_end - AVX_READ; j += AVX_READ) {
                        __m256i br = _mm256_loadu_si256((const __m256i *)&B_real[k*N+j]);
                        __m256i bi = _mm256_loadu_si256((const __m256i *)&B_imag[k*N+j]);
                        __m256i cr = _mm256_loadu_si256((const __m256i *)&C_real[i*N+j]);
                        __m256i ci = _mm256_loadu_si256((const __m256i *)&C_imag[i*N+j]);

                        /*
                            Complex multiplication:
                            A = a + jb
                            B = c + jd

                            A * B = (ac - bd) + j(ad + bc)
                        */
                        __m256i ar_br =
                            _mm256_mullo_epi16(ar, br);
                        __m256i ai_bi =
                            _mm256_mullo_epi16(ai, bi);
                        __m256i ar_bi =
                            _mm256_mullo_epi16(ar, bi);
                        __m256i ai_br =
                            _mm256_mullo_epi16(ai, br);
                        /*
                            Accumulate:
                            C_real += ac - bd
                            C_imag += ad + bc
                        */
                        cr = _mm256_add_epi16(cr, _mm256_sub_epi16(ar_br, ai_bi));
                        ci = _mm256_add_epi16(ci, _mm256_add_epi16(ar_bi, ai_br));

                        _mm256_storeu_si256((__m256i *)&C_real[i*N+j], cr);
                        _mm256_storeu_si256((__m256i *)&C_imag[i*N+j], ci);
                    }
                    for (; j < j_end; j++) {
                        C_real[i*N+j] += A_real[i*K+k] * B_real[k*N+j] - A_imag[i*K+k] * B_imag[k*N+j];
                        C_imag[i*N+j] += A_real[i*K+k] * B_imag[k*N+j] + A_imag[i*K+k] * B_real[k*N+j];
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

static void benchMarkFunction(int M, int N, int K) {
    int16_t *A_real = malloc(M * K * sizeof(int16_t));
    int16_t *A_imag = malloc(M * K * sizeof(int16_t));
    int16_t *B_real = malloc(K * N * sizeof(int16_t));
    int16_t *B_imag = malloc(K * N * sizeof(int16_t));
    int16_t *C_real = malloc(M * N * sizeof(int16_t));
    int16_t *C_imag = malloc(M * N * sizeof(int16_t));
    int16_t *D_real = malloc(M * N * sizeof(int16_t));
    int16_t *D_imag = malloc(M * N * sizeof(int16_t));
    int16_t *E_real = malloc(M * N * sizeof(int16_t));
    int16_t *E_imag = malloc(M * N * sizeof(int16_t));

    if (!A_real || !A_imag ||
        !B_real || !B_imag ||
        !C_real || !C_imag ||
        !D_real || !D_imag ||
        !E_real || !E_imag)
    {
        fprintf(stderr, "Allocation failed\n");
        return;
    }

    for (int i = 0; i < M * K; i++) {
        A_real[i] = 1;
        A_imag[i] = 2;
    }

    for (int i = 0; i < K * N; i++) {
        B_real[i] = 3;
        B_imag[i] = 4;
    }

    double t0 = get_time_sec();
    multiply_ikj_block(
        A_real, A_imag,
        B_real, B_imag,
        C_real, C_imag,
        M, K, N);
    double t1 = get_time_sec();

    double t2 = get_time_sec();
    multiply_ikj_avx_block(
        A_real, A_imag,
        B_real, B_imag,
        D_real, D_imag,
        M, K, N);
    double t3 = get_time_sec();

    printf("\nExecution Time:\n");
    printf("Scalar : %.6f ms\n",
           (t1 - t0) * 1e3);
    printf("AVX2   : %.6f ms\n",
           (t3 - t2) * 1e3);

    bool ok = true;

    for (int i = 0; i < M * N; i++) {
        if (C_real[i] != D_real[i] ||
            C_imag[i] != D_imag[i])
        {
            ok = false;
            printf("Mismatch at %d\n", i);
            printf("Scalar : %d + j%d\n",
                   C_real[i],
                   C_imag[i]);
            printf("AVX2   : %d + j%d\n",
                   D_real[i],
                   D_imag[i]);
            break;
        }
    }
    if (ok)
        printf("Results match\n");

    free(A_real);
    free(A_imag);
    free(B_real);
    free(B_imag);
    free(C_real);
    free(C_imag);
    free(D_real);
    free(D_imag);
    free(E_real);
    free(E_imag);
}

// Isolated Core in CPU
bool pin_me(int corenum)
{
    pthread_t tid = pthread_self();

    cpu_set_t cpuset;

    // Clear all core bit flags
    CPU_ZERO(&cpuset); 
    // Set one core bit flag
    CPU_SET(corenum, &cpuset);

    int ret = pthread_setaffinity_np(
        tid,
        sizeof(cpuset),
        &cpuset);
    
    // Zero return is success
    return ret == 0;
}

int main()
{
    if (!pin_me(1)) {
        perror("pin_me");
        return 1;
    }

    printf("Running on CPU %d\n", sched_getcpu());

    int M = 273 * 12;
    int K = 4;
    int N = 32;

    benchMarkFunction(M, N, K);

    return 0;
}