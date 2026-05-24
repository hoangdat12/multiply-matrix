#include <pthread.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <stdint.h>

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// =========================================================================
// 1. CẤU TRÚC ĐỒNG BỘ THREAD POOL
// =========================================================================
pthread_mutex_t pool_mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  cv_start = PTHREAD_COND_INITIALIZER;
pthread_cond_t  cv_done  = PTHREAD_COND_INITIALIZER;

int tasks_running = 0;
int work_id = 0;       
int shutdown_pool = 0; 

typedef struct {
    const int16_t *A_real; const int16_t *A_imag;
    const int16_t *B_real; const int16_t *B_imag;
    int16_t *C_real;       int16_t *C_imag;
    int M, K, N;
    int start_row;         
    int end_row;           
} ThreadParam;

ThreadParam pool_params[16];
pthread_t pool_threads[16];
int num_pool_threads = 0;

// =========================================================================
// 2. KERNEL AVX2
// =========================================================================
static inline void avx_block(
    int16_t *c,  const int16_t *a,  const int16_t *b,
    int16_t *ci, const int16_t *ai, const int16_t *bi,
    int Mr, int Nr, int Kc, int ldc, int lda, int ldb)
{
    __m256i cr0  = _mm256_loadu_si256((const __m256i *)&c[0 * ldc]);
    __m256i cri0 = _mm256_loadu_si256((const __m256i *)&ci[0 * ldc]);
    __m256i cr1  = _mm256_loadu_si256((const __m256i *)&c[1 * ldc]);
    __m256i cri1 = _mm256_loadu_si256((const __m256i *)&ci[1 * ldc]);
    __m256i cr2  = _mm256_loadu_si256((const __m256i *)&c[2 * ldc]);
    __m256i cri2 = _mm256_loadu_si256((const __m256i *)&ci[2 * ldc]);
    __m256i cr3  = _mm256_loadu_si256((const __m256i *)&c[3 * ldc]);
    __m256i cri3 = _mm256_loadu_si256((const __m256i *)&ci[3 * ldc]);

    for (int k = 0; k < Kc; k++) {
        __m256i br  = _mm256_loadu_si256((const __m256i *)&b[k * ldb]);
        __m256i bii = _mm256_loadu_si256((const __m256i *)&bi[k * ldb]);
        __m256i ar, aii;

        ar  = _mm256_set1_epi16(a[0 * lda + k]);
        aii = _mm256_set1_epi16(ai[0 * lda + k]);
        cr0  = _mm256_add_epi16(cr0,  _mm256_sub_epi16(_mm256_mullo_epi16(ar, br), _mm256_mullo_epi16(aii, bii)));
        cri0 = _mm256_add_epi16(cri0, _mm256_add_epi16(_mm256_mullo_epi16(ar, bii), _mm256_mullo_epi16(aii, br)));

        ar  = _mm256_set1_epi16(a[1 * lda + k]);
        aii = _mm256_set1_epi16(ai[1 * lda + k]);
        cr1  = _mm256_add_epi16(cr1,  _mm256_sub_epi16(_mm256_mullo_epi16(ar, br), _mm256_mullo_epi16(aii, bii)));
        cri1 = _mm256_add_epi16(cri1, _mm256_add_epi16(_mm256_mullo_epi16(ar, bii), _mm256_mullo_epi16(aii, br)));

        ar  = _mm256_set1_epi16(a[2 * lda + k]);
        aii = _mm256_set1_epi16(ai[2 * lda + k]);
        cr2  = _mm256_add_epi16(cr2,  _mm256_sub_epi16(_mm256_mullo_epi16(ar, br), _mm256_mullo_epi16(aii, bii)));
        cri2 = _mm256_add_epi16(cri2, _mm256_add_epi16(_mm256_mullo_epi16(ar, bii), _mm256_mullo_epi16(aii, br)));

        ar  = _mm256_set1_epi16(a[3 * lda + k]);
        aii = _mm256_set1_epi16(ai[3 * lda + k]);
        cr3  = _mm256_add_epi16(cr3,  _mm256_sub_epi16(_mm256_mullo_epi16(ar, br), _mm256_mullo_epi16(aii, bii)));
        cri3 = _mm256_add_epi16(cri3, _mm256_add_epi16(_mm256_mullo_epi16(ar, bii), _mm256_mullo_epi16(aii, br)));
    }

    _mm256_storeu_si256((__m256i *)&c[0 * ldc], cr0);
    _mm256_storeu_si256((__m256i *)&ci[0 * ldc], cri0);
    _mm256_storeu_si256((__m256i *)&c[1 * ldc], cr1);
    _mm256_storeu_si256((__m256i *)&ci[1 * ldc], cri1);
    _mm256_storeu_si256((__m256i *)&c[2 * ldc], cr2);
    _mm256_storeu_si256((__m256i *)&ci[2 * ldc], cri2);
    _mm256_storeu_si256((__m256i *)&c[3 * ldc], cr3);
    _mm256_storeu_si256((__m256i *)&ci[3 * ldc], cri3);
}

static inline void scalar_block(
    int16_t *c,  const int16_t *a,  const int16_t *b,
    int16_t *ci, const int16_t *ai, const int16_t *bi,
    int Mr, int Nr, int Kc, int ldc, int lda, int ldb)
{
    for (int i = 0; i < Mr; i++) {
        for (int k = 0; k < Kc; k++) {
            int16_t ar  = a[i * lda + k];
            int16_t aii = ai[i * lda + k];
            for (int j = 0; j < Nr; j++) {
                int16_t br  = b[k * ldb + j];
                int16_t bii = bi[k * ldb + j];
                c[i * ldc + j]  += ar * br  - aii * bii;
                ci[i * ldc + j] += ar * bii + aii * br;
            }
        }
    }
}

// =========================================================================
// 3. THREAD POOL WORKER & API
// =========================================================================
void* pool_worker_thread(void* arg) {
    ThreadParam* p = (ThreadParam*)arg;
    int my_last_work_id = 0; 
    
    // Đã đồng nhất Mc = 192 (vừa khít 26KB L1 Cache)
    const int Mc = 192, Kc = 4, Nc = 32;
    const int Mr = 4, Nr = 16; 

    while (1) {
        pthread_mutex_lock(&pool_mtx);
        while (my_last_work_id == work_id && shutdown_pool == 0) {
            pthread_cond_wait(&cv_start, &pool_mtx);
        }
        
        if (shutdown_pool) {
            pthread_mutex_unlock(&pool_mtx);
            break; 
        }
        
        my_last_work_id = work_id; 
        pthread_mutex_unlock(&pool_mtx);

        for (int ib = p->start_row; ib < p->end_row; ib += Mc) {
            int i_blk = MIN(Mc, p->end_row - ib);

            memset(&p->C_real[ib * p->N], 0, i_blk * p->N * sizeof(int16_t));
            memset(&p->C_imag[ib * p->N], 0, i_blk * p->N * sizeof(int16_t));

            for (int kb = 0; kb < p->K; kb += Kc) {
                int k_blk = MIN(Kc, p->K - kb);
                for (int jb = 0; jb < p->N; jb += Nc) {
                    int j_blk = MIN(Nc, p->N - jb);

                    const int16_t *ma  = &p->A_real[ib * p->K + kb];
                    const int16_t *mai = &p->A_imag[ib * p->K + kb];
                    const int16_t *mb  = &p->B_real[kb * p->N + jb];
                    const int16_t *mbi = &p->B_imag[kb * p->N + jb];
                    int16_t       *mc  = &p->C_real[ib * p->N + jb];
                    int16_t       *mci = &p->C_imag[ib * p->N + jb];

                    for (int i2 = 0; i2 < i_blk; i2 += Mr) {
                        int mr = MIN(Mr, i_blk - i2);
                        for (int j2 = 0; j2 < j_blk; j2 += Nr) {
                            int nr = MIN(Nr, j_blk - j2);
                            
                            const int16_t *a  = &ma[i2 * p->K];
                            const int16_t *ai = &mai[i2 * p->K];
                            const int16_t *b  = &mb[j2];
                            const int16_t *bi = &mbi[j2];
                            int16_t       *c  = &mc[i2 * p->N + j2];
                            int16_t       *ci = &mci[i2 * p->N + j2];

                            if (mr == Mr && nr == Nr) {
                                avx_block(c, a, b, ci, ai, bi, Mr, Nr, k_blk, p->N, p->K, p->N);
                            } else {
                                scalar_block(c, a, b, ci, ai, bi, mr, nr, k_blk, p->N, p->K, p->N);
                            }
                        }
                    }
                }
            }
        }

        pthread_mutex_lock(&pool_mtx);
        tasks_running--;
        if (tasks_running == 0) {
            pthread_cond_signal(&cv_done); 
        }
        pthread_mutex_unlock(&pool_mtx);
    }
    return NULL; 
}

/**
 * Khởi tạo thread và đặt các biến trạng thái về 0
 * Các thread được tạo ra được đưa vào pool_worker_thread
 *  và vào trạng thái sleep
 */
void init_thread_pool(int num_threads) {
    pthread_mutex_lock(&pool_mtx);
    shutdown_pool = 0;
    tasks_running = 0;
    pthread_mutex_unlock(&pool_mtx);

    num_pool_threads = num_threads;
    for (int t = 0; t < num_threads; t++) {
        pthread_create(&pool_threads[t], NULL, pool_worker_thread, &pool_params[t]);
    }
}

/**
 * Khi hoàn thành, hàm này bật cờ shutdown_pool = 1
 * đánh thức tất cả các luồng đang ngủ (cv_start)
 * và dùng pthread_join để chờ chúng dọn dẹp và thoát hoàn toàn.
 */
void destroy_thread_pool() {
    pthread_mutex_lock(&pool_mtx);
    shutdown_pool = 1;
    pthread_cond_broadcast(&cv_start);
    pthread_mutex_unlock(&pool_mtx);

    for (int t = 0; t < num_pool_threads; t++) {
        pthread_join(pool_threads[t], NULL);
    }
}

/**
 * Hàm này là main thread để cắt nhỏ ma trận và ném cho các worker
 */
void dispatch_slot_to_pool(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real, int16_t *C_imag,
    int M, int K, int N)
{
    // Đã đồng nhất Mc = 192 cho khâu làm tròn chia block
    const int Mc = 192;
    int raw_rows_per_thread = (M + num_pool_threads - 1) / num_pool_threads;
    int rows_per_thread = ((raw_rows_per_thread + Mc - 1) / Mc) * Mc;

    int current_row = 0;
    int active_threads = 0;

    for (int t = 0; t < num_pool_threads; t++) {
        if (current_row >= M) break; 
        
        pool_params[t].A_real = A_real; pool_params[t].A_imag = A_imag;
        pool_params[t].B_real = B_real; pool_params[t].B_imag = B_imag;
        pool_params[t].C_real = C_real; pool_params[t].C_imag = C_imag;
        pool_params[t].M = M; pool_params[t].K = K; pool_params[t].N = N;
        
        pool_params[t].start_row = current_row;
        int next_row = current_row + rows_per_thread;
        pool_params[t].end_row = MIN(next_row, M);
        
        current_row = pool_params[t].end_row;
        active_threads++;
    }

    pthread_mutex_lock(&pool_mtx);
    work_id++;
    tasks_running = active_threads;
    pthread_cond_broadcast(&cv_start); 

    while (tasks_running > 0) {
        pthread_cond_wait(&cv_done, &pool_mtx);
    }
    pthread_mutex_unlock(&pool_mtx);
}

// =========================================================================
// 4. HÀM CHẠY SINGLE THREAD (Baseline)
// =========================================================================
void multiply_ikj_avx_block_single(
    const int16_t *A_real, const int16_t *A_imag,
    const int16_t *B_real, const int16_t *B_imag,
    int16_t *C_real, int16_t *C_imag,
    int M, int K, int N)
{
    // Đã đồng nhất Mc = 192
    const int Mc = 192, Kc = 4, Nc = 32;
    const int Mr = 4, Nr = 16; 

    memset(C_real, 0, M * N * sizeof(*C_real));
    memset(C_imag, 0, M * N * sizeof(*C_imag));

    for (int ib = 0; ib < M; ib += Mc) {
        int i_blk = MIN(Mc, M - ib);
        for (int kb = 0; kb < K; kb += Kc) {
            int k_blk = MIN(Kc, K - kb);
            for (int jb = 0; jb < N; jb += Nc) {
                int j_blk = MIN(Nc, N - jb);
                const int16_t *ma  = &A_real[ib * K + kb];
                const int16_t *mai = &A_imag[ib * K + kb];
                const int16_t *mb  = &B_real[kb * N + jb];
                const int16_t *mbi = &B_imag[kb * N + jb];
                int16_t       *mc  = &C_real[ib * N + jb];
                int16_t       *mci = &C_imag[ib * N + jb];

                for (int i2 = 0; i2 < i_blk; i2 += Mr) {
                    int mr = MIN(Mr, i_blk - i2);
                    for (int j2 = 0; j2 < j_blk; j2 += Nr) {
                        int nr = MIN(Nr, j_blk - j2);
                        const int16_t *a  = &ma[i2 * K];
                        const int16_t *ai = &mai[i2 * K];
                        const int16_t *b  = &mb[j2];
                        const int16_t *bi = &mbi[j2];
                        int16_t       *c  = &mc[i2 * N + j2];
                        int16_t       *ci = &mci[i2 * N + j2];

                        if (mr == Mr && nr == Nr) {
                            avx_block(c, a, b, ci, ai, bi, Mr, Nr, k_blk, N, K, N);
                        } else {
                            scalar_block(c, a, b, ci, ai, bi, mr, nr, k_blk, N, K, N);
                        }
                    }
                }
            }
        }
    }
}

// =========================================================================
// 5. CÁC HÀM TIỆN ÍCH & BENCHMARK MAIN
// =========================================================================
static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void init_matrix(int16_t *mat, int size) {
    for (int i = 0; i < size; i++) mat[i] = (rand() % 10) - 5; 
}

int verify_result(const int16_t *C_ref, const int16_t *C_test, int size) {
    for (int i = 0; i < size; i++) {
        if (C_ref[i] != C_test[i]) return 0;
    }
    return 1;
}

int main() {
    // 1 SLOT 5G
    int M = 45864, K = 4, N = 32;
    int size_A = M * K, size_B = K * N, size_C = M * N;

    printf("=== BENCHMARK THREAD POOL 5G SLOT (Tối ưu 8 Luồng - Mc=192) ===\n");
    printf("Matrix size: %dx%d * %dx%d\n\n", M, K, K, N);
    
    int16_t *A_real = aligned_alloc(32, size_A * sizeof(int16_t));
    int16_t *A_imag = aligned_alloc(32, size_A * sizeof(int16_t));
    int16_t *B_real = aligned_alloc(32, size_B * sizeof(int16_t));
    int16_t *B_imag = aligned_alloc(32, size_B * sizeof(int16_t));
    
    int16_t *C_ref_real = aligned_alloc(32, size_C * sizeof(int16_t));
    int16_t *C_ref_imag = aligned_alloc(32, size_C * sizeof(int16_t));
    int16_t *C_test_real = aligned_alloc(32, size_C * sizeof(int16_t));
    int16_t *C_test_imag = aligned_alloc(32, size_C * sizeof(int16_t));

    srand(time(NULL));
    init_matrix(A_real, size_A); init_matrix(A_imag, size_A);
    init_matrix(B_real, size_B); init_matrix(B_imag, size_B);

    double start, end;

    // --- CHẠY BASELINE (Single Thread) ---
    start = get_time();
    multiply_ikj_avx_block_single(A_real, A_imag, B_real, B_imag, C_ref_real, C_ref_imag, M, K, N);
    end = get_time();
    double time_single = (end - start) * 1000.0;
    printf("[Baseline] Single-thread : %8.3f ms\n", time_single);

    // --- CHẠY THREAD POOL (Fix cứng 8 Luồng) ---
    int threads = 8;
    init_thread_pool(threads);
    
    // Warm-up cache (Chạy mồi)
    dispatch_slot_to_pool(A_real, A_imag, B_real, B_imag, C_test_real, C_test_imag, M, K, N);
    
    // Xóa rác
    memset(C_test_real, 0, size_C * sizeof(int16_t));
    memset(C_test_imag, 0, size_C * sizeof(int16_t));
    
    // Đo thật
    start = get_time();
    dispatch_slot_to_pool(A_real, A_imag, B_real, B_imag, C_test_real, C_test_imag, M, K, N);
    end = get_time();
    
    double time_multi = (end - start) * 1000.0;
    double speedup = time_single / time_multi;

    int is_correct = verify_result(C_ref_real, C_test_real, size_C) && 
                     verify_result(C_ref_imag, C_test_imag, size_C);

    printf("[ThreadPool] %2d threads  : %8.3f ms | Speedup: %4.2fx | Correct: %s %s\n", 
           threads, time_multi, speedup, is_correct ? "YES" : "NO!", 
           (time_multi < 0.5) ? " [ <0.5ms ]" : "");

    destroy_thread_pool();

    free(A_real); free(A_imag); free(B_real); free(B_imag);
    free(C_ref_real); free(C_ref_imag); free(C_test_real); free(C_test_imag);

    return 0;
}