#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdint.h>
#include <immintrin.h>
#include <stdbool.h>
#include <string.h>
#include <pthread.h>

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
 *   - Thread Pool    : Chia M hàng cho nhiều thread, mỗi thread chạy AVX2 + blocking
 *
 * Tham số có thể truyền lúc build (gcc -D...) hoặc để mặc định:
 *   -DM_DIM=3276  -DK_DIM=4  -DN_DIM=32
 *   -DMC_VALUE=192  -DKC_VALUE=4  -DNC_VALUE=32
 *   -DNUM_THREAD=8
 *
 * Build: gcc -O3 -mavx2 -march=native -lpthread -o build/multiplyMatrix multiplyMatrix.c
 *        gcc -O3 -mavx2 -march=native -lpthread -DM_DIM=3276 -DK_DIM=4 -DN_DIM=32 \
 *            -DMC_VALUE=192 -DKC_VALUE=4 -DNC_VALUE=32 -DNUM_THREAD=8 \
 *            -o build/multiplyMatrix multiplyMatrix.c
 */

#define MIN(a, b) (((a) < (b)) ? (a) : (b))

/* ─────────────────────────────────────────────────────────────
 * Tham số mặc định – override bằng -D khi build
 * ───────────────────────────────────────────────────────────── */
#ifndef M_DIM
#define M_DIM      (273 * 12 * 14)   /* 3276 subcarriers */
#endif

#ifndef K_DIM
#define K_DIM      4                 /* rank */
#endif

#ifndef N_DIM
#define N_DIM      32                /* số antenna ports */
#endif

/* Mc = 192:
 *   Mỗi thread giữ 192 hàng × 4 cột × 2B = 1536B (A)
 *   + B toàn bộ: 4×32×2×2 = 512B (real+imag)
 *   + C: 192×32×2×2 = 24576B
 *   Tổng ≈ 26KB → vừa khít L1 cache (32KB)
 */
#ifndef MC_VALUE
#define MC_VALUE   192
#endif

#ifndef KC_VALUE
#define KC_VALUE   4
#endif

#ifndef NC_VALUE
#define NC_VALUE   32
#endif

#ifndef NUM_THREAD
#define NUM_THREAD 8
#endif

/* ─────────────────────────────────────────────────────────────
 * Thread pool globals
 * ───────────────────────────────────────────────────────────── */
pthread_mutex_t pool_mtx  = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  cv_start  = PTHREAD_COND_INITIALIZER;
pthread_cond_t  cv_done   = PTHREAD_COND_INITIALIZER;

static int tasks_running  = 0;
static int work_id        = 0;
static int shutdown_pool  = 0;

/* Tham số runtime (đọc từ macro, có thể mở rộng thành argv sau) */
static int g_Mc         = MC_VALUE;
static int g_Kc         = KC_VALUE;
static int g_Nc         = NC_VALUE;
static int g_num_thread = NUM_THREAD;

typedef struct {
    const int16_t *A_real; const int16_t *A_imag;
    const int16_t *B_real; const int16_t *B_imag;
    int16_t       *C_real; int16_t       *C_imag;
    int M, K, N;
    int start_row;
    int end_row;
} ThreadParam;

#define MAX_THREADS 64
static ThreadParam pool_params[MAX_THREADS];
static pthread_t   pool_threads[MAX_THREADS];
static int         num_pool_threads = 0;

/* ─────────────────────────────────────────────────────────────
 * 1. multiply_ijk  –  baseline (cache-unfriendly)
 *
 * Không có cache friendly vì khi đọc ma trận B theo cột
 * cache sẽ miss liên tục và phải ra RAM để đọc dữ liệu
 * ───────────────────────────────────────────────────────────── */
void multiply_ijk(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real, int16_t *C_imag,
    int M, int K, int N)
{
    memset(C_real, 0, M * N * sizeof(*C_real));
    memset(C_imag, 0, M * N * sizeof(*C_imag));

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            int32_t sum_real = 0, sum_imag = 0;
            for (int k = 0; k < K; k++) {
                int16_t ar = A_real[i*K + k], ai = A_imag[i*K + k];
                int16_t br = B_real[k*N + j], bi = B_imag[k*N + j];
                sum_real += (int32_t)ar*br - (int32_t)ai*bi;
                sum_imag += (int32_t)ar*bi + (int32_t)ai*br;
            }
            C_real[i*N + j] = (int16_t)sum_real;
            C_imag[i*N + j] = (int16_t)sum_imag;
        }
    }
}

/* ─────────────────────────────────────────────────────────────
 * 2. multiply_ikj  –  loop reorder (cache-friendly)
 *
 * A[i][k] đưa ra ngoài loop j → load 1 lần, dùng N lần
 * B[k][j] và C[i][j] access tuần tự theo j → cache line hit
 * ───────────────────────────────────────────────────────────── */
void multiply_ikj(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real, int16_t *C_imag,
    int M, int K, int N)
{
    memset(C_real, 0, M * N * sizeof(*C_real));
    memset(C_imag, 0, M * N * sizeof(*C_imag));

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            int16_t ar = A_real[i*K + k];
            int16_t ai = A_imag[i*K + k];
            for (int j = 0; j < N; j++) {
                int16_t br = B_real[k*N + j];
                int16_t bi = B_imag[k*N + j];
                C_real[i*N + j] += ar*br - ai*bi;
                C_imag[i*N + j] += ar*bi + ai*br;
            }
        }
    }
}

/* ─────────────────────────────────────────────────────────────
 * 3. multiply_ikj_block  –  cache blocking
 *
 * Chia M thành block Mc để A-block fit L1 cache.
 * ───────────────────────────────────────────────────────────── */
void multiply_ikj_block(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real, int16_t *C_imag,
    int M, int K, int N)
{
    const int Mc = g_Mc, Kc = g_Kc, Nc = g_Nc;

    memset(C_real, 0, M * N * sizeof(*C_real));
    memset(C_imag, 0, M * N * sizeof(*C_imag));

    for (int ib = 0; ib < M; ib += Mc) {
        int i_blk = MIN(Mc, M - ib);
        for (int kb = 0; kb < K; kb += Kc) {
            int k_blk = MIN(Kc, K - kb);
            for (int jb = 0; jb < N; jb += Nc) {
                int j_blk = MIN(Nc, N - jb);

                const int16_t *ma  = &A_real[ib*K + kb];
                const int16_t *mai = &A_imag[ib*K + kb];
                const int16_t *mb  = &B_real[kb*N + jb];
                const int16_t *mbi = &B_imag[kb*N + jb];
                int16_t       *mc  = &C_real[ib*N + jb];
                int16_t       *mci = &C_imag[ib*N + jb];

                for (int i2 = 0; i2 < i_blk; i2++) {
                    for (int k2 = 0; k2 < k_blk; k2++) {
                        int16_t ar  = ma [i2*K + k2];
                        int16_t aii = mai[i2*K + k2];
                        for (int j2 = 0; j2 < j_blk; j2++) {
                            mc [i2*N + j2] += ar *mb [k2*N + j2] - aii*mbi[k2*N + j2];
                            mci[i2*N + j2] += ar *mbi[k2*N + j2] + aii*mb [k2*N + j2];
                        }
                    }
                }
            }
        }
    }
}

/* ─────────────────────────────────────────────────────────────
 * 4. multiply_ikj_avx  –  AVX2, không blocking
 *
 * ar, aii broadcast → _mm256_set1_epi16 (1 YMM cho 16 phần tử)
 * Loop j xử lý 16 phần tử/lần; scalar tail cho phần dư.
 * Với N=32 → loop j chạy đúng 2 lần, không có phần dư.
 * ───────────────────────────────────────────────────────────── */
void multiply_ikj_avx(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real, int16_t *C_imag,
    int M, int K, int N)
{
    const int Nr = 16;

    memset(C_real, 0, M * N * sizeof(*C_real));
    memset(C_imag, 0, M * N * sizeof(*C_imag));

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            __m256i ar  = _mm256_set1_epi16(A_real[i*K + k]);
            __m256i aii = _mm256_set1_epi16(A_imag[i*K + k]);

            int j = 0;
            for (; j <= N - Nr; j += Nr) {
                __m256i br = _mm256_loadu_si256((const __m256i *)&B_real[k*N + j]);
                __m256i bi = _mm256_loadu_si256((const __m256i *)&B_imag[k*N + j]);
                __m256i cr = _mm256_loadu_si256((const __m256i *)&C_real[i*N + j]);
                __m256i ci = _mm256_loadu_si256((const __m256i *)&C_imag[i*N + j]);

                cr = _mm256_add_epi16(cr, _mm256_sub_epi16(
                         _mm256_mullo_epi16(ar,  br),
                         _mm256_mullo_epi16(aii, bi)));
                ci = _mm256_add_epi16(ci, _mm256_add_epi16(
                         _mm256_mullo_epi16(ar,  bi),
                         _mm256_mullo_epi16(aii, br)));

                _mm256_storeu_si256((__m256i *)&C_real[i*N + j], cr);
                _mm256_storeu_si256((__m256i *)&C_imag[i*N + j], ci);
            }
            /* scalar tail khi N không chia hết 16 */
            for (; j < N; j++) {
                int16_t a_r = A_real[i*K+k], a_i = A_imag[i*K+k];
                int16_t b_r = B_real[k*N+j], b_i = B_imag[k*N+j];
                C_real[i*N+j] += a_r*b_r - a_i*b_i;
                C_imag[i*N+j] += a_r*b_i + a_i*b_r;
            }
        }
    }
}

/* ─────────────────────────────────────────────────────────────
 * Micro-kernel: AVX2 cho tile cố định Mr=4, Nr=16
 *
 * Load C vào 8 YMM register (cr0..cr3, cri0..cri3).
 * Accumulate qua toàn bộ Kc trước khi store → giảm
 * số lần load/store C so với multiply_ikj_avx.
 * ───────────────────────────────────────────────────────────── */
static inline void avx_block(
    int16_t *c,  const int16_t *a,  const int16_t *b,
    int16_t *ci, const int16_t *ai, const int16_t *bi,
    int Mr, int Nr, int Kc, int ldc, int lda, int ldb)
{
    (void)Mr; (void)Nr; /* luôn được gọi với Mr=4, Nr=16 */

    __m256i cr0  = _mm256_loadu_si256((const __m256i *)&c [0*ldc]);
    __m256i cri0 = _mm256_loadu_si256((const __m256i *)&ci[0*ldc]);
    __m256i cr1  = _mm256_loadu_si256((const __m256i *)&c [1*ldc]);
    __m256i cri1 = _mm256_loadu_si256((const __m256i *)&ci[1*ldc]);
    __m256i cr2  = _mm256_loadu_si256((const __m256i *)&c [2*ldc]);
    __m256i cri2 = _mm256_loadu_si256((const __m256i *)&ci[2*ldc]);
    __m256i cr3  = _mm256_loadu_si256((const __m256i *)&c [3*ldc]);
    __m256i cri3 = _mm256_loadu_si256((const __m256i *)&ci[3*ldc]);

    for (int k = 0; k < Kc; k++) {
        __m256i br  = _mm256_loadu_si256((const __m256i *)&b [k*ldb]);
        __m256i bii = _mm256_loadu_si256((const __m256i *)&bi[k*ldb]);
        __m256i ar, aii;

#define ACCUM(row, crN, criN) \
        ar  = _mm256_set1_epi16(a [row*lda + k]); \
        aii = _mm256_set1_epi16(ai[row*lda + k]); \
        crN  = _mm256_add_epi16(crN,  _mm256_sub_epi16( \
                   _mm256_mullo_epi16(ar,  br),  _mm256_mullo_epi16(aii, bii))); \
        criN = _mm256_add_epi16(criN, _mm256_add_epi16( \
                   _mm256_mullo_epi16(ar,  bii), _mm256_mullo_epi16(aii, br)));

        ACCUM(0, cr0,  cri0)
        ACCUM(1, cr1,  cri1)
        ACCUM(2, cr2,  cri2)
        ACCUM(3, cr3,  cri3)
#undef  ACCUM
    }

    _mm256_storeu_si256((__m256i *)&c [0*ldc], cr0);
    _mm256_storeu_si256((__m256i *)&ci[0*ldc], cri0);
    _mm256_storeu_si256((__m256i *)&c [1*ldc], cr1);
    _mm256_storeu_si256((__m256i *)&ci[1*ldc], cri1);
    _mm256_storeu_si256((__m256i *)&c [2*ldc], cr2);
    _mm256_storeu_si256((__m256i *)&ci[2*ldc], cri2);
    _mm256_storeu_si256((__m256i *)&c [3*ldc], cr3);
    _mm256_storeu_si256((__m256i *)&ci[3*ldc], cri3);
}

/* Scalar fallback cho tile không đủ Mr=4 hoặc Nr=16 */
static inline void scalar_block(
    int16_t *c,  const int16_t *a,  const int16_t *b,
    int16_t *ci, const int16_t *ai, const int16_t *bi,
    int Mr, int Nr, int Kc, int ldc, int lda, int ldb)
{
    for (int i = 0; i < Mr; i++) {
        for (int k = 0; k < Kc; k++) {
            int16_t ar  = a [i*lda + k];
            int16_t aii = ai[i*lda + k];
            for (int j = 0; j < Nr; j++) {
                c [i*ldc + j] += ar *b [k*ldb + j] - aii*bi[k*ldb + j];
                ci[i*ldc + j] += ar *bi[k*ldb + j] + aii*b [k*ldb + j];
            }
        }
    }
}

/* ─────────────────────────────────────────────────────────────
 * 5. multiply_ikj_avx_block  –  AVX2 + cache blocking (single-thread)
 *
 * Tầng 1 – Cache block (Mc, Kc, Nc từ g_Mc/g_Kc/g_Nc):
 *   Chia M thành block vừa khít L1.
 *
 * Tầng 2 – Micro-tile (Mr=4, Nr=16):
 *   avx_block load C 1 lần, accumulate Kc vòng, store 1 lần.
 *   scalar_block cho phần dư (Mr < 4 hoặc Nr < 16).
 * ───────────────────────────────────────────────────────────── */
void multiply_ikj_avx_block(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real, int16_t *C_imag,
    int M, int K, int N)
{
    const int Mc = g_Mc, Kc = g_Kc, Nc = g_Nc;
    const int Mr = 4, Nr = 16;

    memset(C_real, 0, M * N * sizeof(*C_real));
    memset(C_imag, 0, M * N * sizeof(*C_imag));

    for (int ib = 0; ib < M; ib += Mc) {
        int i_blk = MIN(Mc, M - ib);

        for (int kb = 0; kb < K; kb += Kc) {
            int k_blk = MIN(Kc, K - kb);

            for (int jb = 0; jb < N; jb += Nc) {
                int j_blk = MIN(Nc, N - jb);

                const int16_t *ma  = &A_real[ib*K + kb];
                const int16_t *mai = &A_imag[ib*K + kb];
                const int16_t *mb  = &B_real[kb*N + jb];
                const int16_t *mbi = &B_imag[kb*N + jb];
                int16_t       *mc  = &C_real[ib*N + jb];
                int16_t       *mci = &C_imag[ib*N + jb];

                for (int i2 = 0; i2 < i_blk; i2 += Mr) {
                    int mr = MIN(Mr, i_blk - i2);

                    for (int j2 = 0; j2 < j_blk; j2 += Nr) {
                        int nr = MIN(Nr, j_blk - j2);

                        const int16_t *a  = &ma [i2*K];
                        const int16_t *ai = &mai[i2*K];
                        const int16_t *b  = &mb [j2];
                        const int16_t *bi = &mbi[j2];
                        int16_t       *c  = &mc [i2*N + j2];
                        int16_t       *ci = &mci[i2*N + j2];

                        if (mr == Mr && nr == Nr)
                            avx_block (c, a, b, ci, ai, bi, Mr, Nr, k_blk, N, K, N);
                        else
                            scalar_block(c, a, b, ci, ai, bi, mr, nr, k_blk, N, K, N);
                    }
                }
            }
        }
    }
}

/* ─────────────────────────────────────────────────────────────
 * Thread worker: AVX2 + blocking
 * ───────────────────────────────────────────────────────────── */
static void* pool_worker_thread(void *arg)
{
    ThreadParam *p       = (ThreadParam *)arg;
    int my_last = 0;
    const int Mr = 4, Nr = 16;

    while (1) {
        pthread_mutex_lock(&pool_mtx);
        while (my_last == work_id && !shutdown_pool)
            pthread_cond_wait(&cv_start, &pool_mtx);
        if (shutdown_pool) { pthread_mutex_unlock(&pool_mtx); break; }
        my_last = work_id;
        pthread_mutex_unlock(&pool_mtx);

        const int Mc = g_Mc, Kc = g_Kc, Nc = g_Nc;

        for (int ib = p->start_row; ib < p->end_row; ib += Mc) {
            int i_blk = MIN(Mc, p->end_row - ib);

            memset(&p->C_real[ib * p->N], 0, i_blk * p->N * sizeof(int16_t));
            memset(&p->C_imag[ib * p->N], 0, i_blk * p->N * sizeof(int16_t));

            for (int kb = 0; kb < p->K; kb += Kc) {
                int k_blk = MIN(Kc, p->K - kb);
                for (int jb = 0; jb < p->N; jb += Nc) {
                    int j_blk = MIN(Nc, p->N - jb);

                    const int16_t *ma  = &p->A_real[ib*p->K + kb];
                    const int16_t *mai = &p->A_imag[ib*p->K + kb];
                    const int16_t *mb  = &p->B_real[kb*p->N + jb];
                    const int16_t *mbi = &p->B_imag[kb*p->N + jb];
                    int16_t       *mc  = &p->C_real[ib*p->N + jb];
                    int16_t       *mci = &p->C_imag[ib*p->N + jb];

                    for (int i2 = 0; i2 < i_blk; i2 += Mr) {
                        int mr = MIN(Mr, i_blk - i2);
                        for (int j2 = 0; j2 < j_blk; j2 += Nr) {
                            int nr = MIN(Nr, j_blk - j2);

                            const int16_t *a  = &ma [i2*p->K];
                            const int16_t *ai = &mai[i2*p->K];
                            const int16_t *b  = &mb [j2];
                            const int16_t *bi = &mbi[j2];
                            int16_t       *c  = &mc [i2*p->N + j2];
                            int16_t       *ci = &mci[i2*p->N + j2];

                            if (mr == Mr && nr == Nr)
                                avx_block (c, a, b, ci, ai, bi, Mr, Nr, k_blk, p->N, p->K, p->N);
                            else
                                scalar_block(c, a, b, ci, ai, bi, mr, nr, k_blk, p->N, p->K, p->N);
                        }
                    }
                }
            }
        }

        pthread_mutex_lock(&pool_mtx);
        if (--tasks_running == 0)
            pthread_cond_signal(&cv_done);
        pthread_mutex_unlock(&pool_mtx);
    }
    return NULL;
}

/* ─────────────────────────────────────────────────────────────
 * Thread worker: AVX2, không blocking
 * ───────────────────────────────────────────────────────────── */
static void* pool_worker_no_block(void *arg)
{
    ThreadParam *p       = (ThreadParam *)arg;
    int          my_last = 0;
    const int    Nr      = 16;

    while (1) {
        pthread_mutex_lock(&pool_mtx);
        while (my_last == work_id && !shutdown_pool)
            pthread_cond_wait(&cv_start, &pool_mtx);
        if (shutdown_pool) { pthread_mutex_unlock(&pool_mtx); break; }
        my_last = work_id;
        pthread_mutex_unlock(&pool_mtx);

        int rows = p->end_row - p->start_row;
        memset(&p->C_real[p->start_row * p->N], 0, rows * p->N * sizeof(int16_t));
        memset(&p->C_imag[p->start_row * p->N], 0, rows * p->N * sizeof(int16_t));

        for (int i = p->start_row; i < p->end_row; i++) {
            for (int k = 0; k < p->K; k++) {
                __m256i ar  = _mm256_set1_epi16(p->A_real[i*p->K + k]);
                __m256i aii = _mm256_set1_epi16(p->A_imag[i*p->K + k]);

                int j = 0;
                for (; j <= p->N - Nr; j += Nr) {
                    __m256i br = _mm256_loadu_si256((const __m256i *)&p->B_real[k*p->N + j]);
                    __m256i bi = _mm256_loadu_si256((const __m256i *)&p->B_imag[k*p->N + j]);
                    __m256i cr = _mm256_loadu_si256((const __m256i *)&p->C_real[i*p->N + j]);
                    __m256i ci = _mm256_loadu_si256((const __m256i *)&p->C_imag[i*p->N + j]);

                    cr = _mm256_add_epi16(cr, _mm256_sub_epi16(
                             _mm256_mullo_epi16(ar,  br), _mm256_mullo_epi16(aii, bi)));
                    ci = _mm256_add_epi16(ci, _mm256_add_epi16(
                             _mm256_mullo_epi16(ar,  bi), _mm256_mullo_epi16(aii, br)));

                    _mm256_storeu_si256((__m256i *)&p->C_real[i*p->N + j], cr);
                    _mm256_storeu_si256((__m256i *)&p->C_imag[i*p->N + j], ci);
                }
                for (; j < p->N; j++) {
                    int16_t a_r = p->A_real[i*p->K+k], a_i = p->A_imag[i*p->K+k];
                    int16_t b_r = p->B_real[k*p->N+j], b_i = p->B_imag[k*p->N+j];
                    p->C_real[i*p->N+j] += a_r*b_r - a_i*b_i;
                    p->C_imag[i*p->N+j] += a_r*b_i + a_i*b_r;
                }
            }
        }

        pthread_mutex_lock(&pool_mtx);
        if (--tasks_running == 0)
            pthread_cond_signal(&cv_done);
        pthread_mutex_unlock(&pool_mtx);
    }
    return NULL;
}

/* ─────────────────────────────────────────────────────────────
 * Thread pool: init / destroy / dispatch
 * ───────────────────────────────────────────────────────────── */
static void _init_pool(int num_threads, void *(*worker)(void *))
{
    pthread_mutex_lock(&pool_mtx);
    shutdown_pool = 0;
    tasks_running = 0;
    work_id       = 0;
    pthread_mutex_unlock(&pool_mtx);

    num_pool_threads = num_threads;
    for (int t = 0; t < num_threads; t++)
        pthread_create(&pool_threads[t], NULL, worker, &pool_params[t]);
}

void init_thread_pool_blocking(int num_threads) {
    _init_pool(num_threads, pool_worker_thread);
}

void init_thread_pool_no_block(int num_threads) {
    _init_pool(num_threads, pool_worker_no_block);
}

void destroy_thread_pool(void)
{
    pthread_mutex_lock(&pool_mtx);
    shutdown_pool = 1;
    pthread_cond_broadcast(&cv_start);
    pthread_mutex_unlock(&pool_mtx);

    for (int t = 0; t < num_pool_threads; t++)
        pthread_join(pool_threads[t], NULL);

    num_pool_threads = 0;
}

static void _dispatch(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real,       int16_t *C_imag,
    int M, int K, int N,
    int rows_per_thread)
{
    int current_row    = 0;
    int active_threads = 0;

    for (int t = 0; t < num_pool_threads; t++) {
        if (current_row >= M) break;

        pool_params[t].A_real = A_real; pool_params[t].A_imag = A_imag;
        pool_params[t].B_real = B_real; pool_params[t].B_imag = B_imag;
        pool_params[t].C_real = C_real; pool_params[t].C_imag = C_imag;
        pool_params[t].M = M; pool_params[t].K = K; pool_params[t].N = N;
        pool_params[t].start_row = current_row;
        pool_params[t].end_row   = MIN(current_row + rows_per_thread, M);
        current_row = pool_params[t].end_row;
        active_threads++;
    }

    pthread_mutex_lock(&pool_mtx);
    work_id++;
    tasks_running = active_threads;
    pthread_cond_broadcast(&cv_start);
    while (tasks_running > 0)
        pthread_cond_wait(&cv_done, &pool_mtx);
    pthread_mutex_unlock(&pool_mtx);
}

void dispatch_slot_to_pool(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real,       int16_t *C_imag,
    int M, int K, int N)
{
    int raw  = (M + num_pool_threads - 1) / num_pool_threads;
    int rows = ((raw + g_Mc - 1) / g_Mc) * g_Mc;
    _dispatch(A_real, A_imag, B_real, B_imag, C_real, C_imag, M, K, N, rows);
}

void dispatch_no_block_to_pool(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real,       int16_t *C_imag,
    int M, int K, int N)
{
    int rows = (M + num_pool_threads - 1) / num_pool_threads;
    _dispatch(A_real, A_imag, B_real, B_imag, C_real, C_imag, M, K, N, rows);
}

/* ─────────────────────────────────────────────────────────────
 * Helpers
 * ───────────────────────────────────────────────────────────── */
static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static bool verify(const int16_t *ref_r, const int16_t *ref_i,
                   const int16_t *got_r, const int16_t *got_i,
                   int MN, const char *name)
{
    for (int i = 0; i < MN; i++) {
        if (ref_r[i] != got_r[i] || ref_i[i] != got_i[i]) {
            printf("  [FAIL] %s  tại index %d: ref=(%d,%d) got=(%d,%d)\n",
                   name, i, ref_r[i], ref_i[i], got_r[i], got_i[i]);
            return false;
        }
    }
    return true;
}

/* ─────────────────────────────────────────────────────────────
 * Benchmark
 * ───────────────────────────────────────────────────────────── */
static void benchMarkFunction(int M, int K, int N)
{
    int16_t *A_real = malloc(M * K * sizeof(int16_t));
    int16_t *A_imag = malloc(M * K * sizeof(int16_t));
    int16_t *B_real = malloc(K * N * sizeof(int16_t));
    int16_t *B_imag = malloc(K * N * sizeof(int16_t));

#define ALLOC2(name) \
    int16_t *Out_##name##_real = calloc(M * N, sizeof(int16_t)); \
    int16_t *Out_##name##_imag = calloc(M * N, sizeof(int16_t))

    ALLOC2(ijk);
    ALLOC2(ikj);
    ALLOC2(block);
    ALLOC2(avx_nb);
    ALLOC2(avx_blk);
    ALLOC2(pool_blk);
    ALLOC2(pool_nb);
#undef ALLOC2

    for (int i = 0; i < M*K; i++) { A_real[i] = 1; A_imag[i] = 2; }
    for (int i = 0; i < K*N; i++) { B_real[i] = 3; B_imag[i] = 4; }

    double t0, t1;

#define BENCH(fn, out) \
    t0 = get_time_sec(); \
    fn(A_real, A_imag, B_real, B_imag, Out_##out##_real, Out_##out##_imag, M, K, N); \
    t1 = get_time_sec();

    BENCH(multiply_ijk,           ijk)     double ms_ijk     = (t1-t0)*1e3;
    BENCH(multiply_ikj,           ikj)     double ms_ikj     = (t1-t0)*1e3;
    BENCH(multiply_ikj_block,     block)   double ms_block   = (t1-t0)*1e3;
    BENCH(multiply_ikj_avx,       avx_nb)  double ms_avx_nb  = (t1-t0)*1e3;
    BENCH(multiply_ikj_avx_block, avx_blk) double ms_avx_blk = (t1-t0)*1e3;
#undef BENCH

    init_thread_pool_blocking(g_num_thread);
    t0 = get_time_sec();
    dispatch_slot_to_pool(A_real, A_imag, B_real, B_imag, Out_pool_blk_real, Out_pool_blk_imag, M, K, N);
    t1 = get_time_sec();
    double ms_pool_blk = (t1-t0)*1e3;
    destroy_thread_pool();

    init_thread_pool_no_block(g_num_thread);
    t0 = get_time_sec();
    dispatch_no_block_to_pool(A_real, A_imag, B_real, B_imag, Out_pool_nb_real, Out_pool_nb_imag, M, K, N);
    t1 = get_time_sec();
    double ms_pool_nb = (t1-t0)*1e3;
    destroy_thread_pool();

    printf("\n=== Benchmark  M=%d  K=%d  N=%d  Mc=%d  Kc=%d  Nc=%d  threads=%d ===\n",
           M, K, N, g_Mc, g_Kc, g_Nc, g_num_thread);
    printf("%-30s %8.4f ms  (baseline)\n",              "1. Basic ijk",            ms_ijk);
    printf("%-30s %8.4f ms  (%.1fx)\n",                 "2. Basic ikj",            ms_ikj,     ms_ijk/ms_ikj);
    printf("%-30s %8.4f ms  (%.1fx)\n",                 "3. Block ikj",            ms_block,   ms_ijk/ms_block);
    printf("%-30s %8.4f ms  (%.1fx)\n",                 "4. AVX2 No-Block ikj",    ms_avx_nb,  ms_ijk/ms_avx_nb);
    printf("%-30s %8.4f ms  (%.1fx)\n",                 "5. AVX2 Block ikj",       ms_avx_blk, ms_ijk/ms_avx_blk);
    printf("%-30s %8.4f ms  (%.1fx)  [%d threads]\n",   "6. ThreadPool AVX2+Block",ms_pool_blk,ms_ijk/ms_pool_blk, g_num_thread);
    printf("%-30s %8.4f ms  (%.1fx)  [%d threads]\n",   "7. ThreadPool AVX2 NoBl", ms_pool_nb, ms_ijk/ms_pool_nb,  g_num_thread);

    printf("\n--- Kiểm tra kết quả (so với ijk) ---\n");
    int MN = M * N;
    bool all_ok = true;
    all_ok &= verify(Out_ijk_real, Out_ijk_imag, Out_ikj_real,      Out_ikj_imag,      MN, "ikj");
    all_ok &= verify(Out_ijk_real, Out_ijk_imag, Out_block_real,    Out_block_imag,    MN, "block");
    all_ok &= verify(Out_ijk_real, Out_ijk_imag, Out_avx_nb_real,   Out_avx_nb_imag,   MN, "avx_no_block");
    all_ok &= verify(Out_ijk_real, Out_ijk_imag, Out_avx_blk_real,  Out_avx_blk_imag,  MN, "avx_block");
    all_ok &= verify(Out_ijk_real, Out_ijk_imag, Out_pool_blk_real, Out_pool_blk_imag, MN, "pool_block");
    all_ok &= verify(Out_ijk_real, Out_ijk_imag, Out_pool_nb_real,  Out_pool_nb_imag,  MN, "pool_no_block");

    if (all_ok)
        printf("=> Tất cả thuật toán cho kết quả đúng.\n");

    printf("\nC[0][0] = %d + %dj\n", Out_ijk_real[0], Out_ijk_imag[0]);

    free(A_real); free(A_imag); free(B_real); free(B_imag);
    free(Out_ijk_real);  free(Out_ijk_imag);
    free(Out_ikj_real);  free(Out_ikj_imag);
    free(Out_block_real);    free(Out_block_imag);
    free(Out_avx_nb_real);   free(Out_avx_nb_imag);
    free(Out_avx_blk_real);  free(Out_avx_blk_imag);
    free(Out_pool_blk_real); free(Out_pool_blk_imag);
    free(Out_pool_nb_real);  free(Out_pool_nb_imag);
}

/* ─────────────────────────────────────────────────────────────
 * main – đọc tham số từ macro (truyền lúc build bằng -D)
 * ───────────────────────────────────────────────────────────── */
int main(void)
{
    int M = M_DIM;
    int K = K_DIM;
    int N = N_DIM;

    g_Mc         = MC_VALUE;
    g_Kc         = KC_VALUE;
    g_Nc         = NC_VALUE;
    g_num_thread = NUM_THREAD;

    printf("Config: M=%d  K=%d  N=%d  Mc=%d  Kc=%d  Nc=%d  threads=%d\n",
           M, K, N, g_Mc, g_Kc, g_Nc, g_num_thread);

    benchMarkFunction(M, K, N);
    return 0;
}