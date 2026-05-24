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
// THREAD POOL & SYNC
// =========================================================================
pthread_mutex_t pool_mtx = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t  cv_start = PTHREAD_COND_INITIALIZER;
pthread_cond_t  cv_done  = PTHREAD_COND_INITIALIZER;
int tasks_running = 0, work_id = 0, shutdown_pool = 0;

typedef struct {
    const int16_t *A_r, *A_i, *B_r, *B_i;
    int16_t *C_r, *C_i;
    int M, K, N, Mc, start, end;
} ThreadParam;

ThreadParam pool_params[16];
pthread_t pool_threads[16];
int num_threads = 6; 

static inline double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

// =========================================================================
// KERNEL AVX2
// =========================================================================
static inline void avx_block(int16_t *c, const int16_t *a, const int16_t *b, int16_t *ci, const int16_t *ai, const int16_t *bi, int Mr, int Kc, int ldc, int lda, int ldb) {
    for (int i = 0; i < Mr; i++) {
        __m256i cr = _mm256_loadu_si256((const __m256i *)&c[i * ldc]), cri = _mm256_loadu_si256((const __m256i *)&ci[i * ldc]);
        for (int k = 0; k < Kc; k++) {
            __m256i ar = _mm256_set1_epi16(a[i * lda + k]), aii = _mm256_set1_epi16(ai[i * lda + k]);
            __m256i br = _mm256_loadu_si256((const __m256i *)&b[k * ldb]), bii = _mm256_loadu_si256((const __m256i *)&bi[k * ldb]);
            cr  = _mm256_add_epi16(cr,  _mm256_sub_epi16(_mm256_mullo_epi16(ar, br), _mm256_mullo_epi16(aii, bii)));
            cri = _mm256_add_epi16(cri, _mm256_add_epi16(_mm256_mullo_epi16(ar, bii), _mm256_mullo_epi16(aii, br)));
        }
        _mm256_storeu_si256((__m256i *)&c[i * ldc], cr); _mm256_storeu_si256((__m256i *)&ci[i * ldc], cri);
    }
}

// =========================================================================
// WORKER THREAD
// =========================================================================
void* worker(void* arg) {
    ThreadParam* p = (ThreadParam*)arg;
    int my_id = 0;
    while (1) {
        pthread_mutex_lock(&pool_mtx);
        while (my_id == work_id && !shutdown_pool) pthread_cond_wait(&cv_start, &pool_mtx);
        if (shutdown_pool) { pthread_mutex_unlock(&pool_mtx); break; }
        my_id = work_id; pthread_mutex_unlock(&pool_mtx);

        int Mc = p->Mc, Kc = 4, Mr = 4, Nr = 16;
        for (int ib = p->start; ib < p->end; ib += Mc) {
            int i_blk = MIN(Mc, p->end - ib);
            memset(&p->C_r[ib * p->N], 0, i_blk * p->N * sizeof(int16_t));
            memset(&p->C_i[ib * p->N], 0, i_blk * p->N * sizeof(int16_t));
            for (int kb = 0; kb < p->K; kb += Kc)
                for (int i2 = 0; i2 < i_blk; i2 += Mr)
                    for (int j2 = 0; j2 < p->N; j2 += Nr)
                        avx_block(&p->C_r[ib*p->N + i2*p->N + j2], &p->A_r[ib*p->K + i2*p->K + kb], &p->B_r[kb*p->N + j2],
                                  &p->C_i[ib*p->N + i2*p->N + j2], &p->A_i[ib*p->K + i2*p->K + kb], &p->B_i[kb*p->N + j2], Mr, Kc, p->N, p->K, p->N);
        }
        pthread_mutex_lock(&pool_mtx);
        if (--tasks_running == 0) pthread_cond_signal(&cv_done);
        pthread_mutex_unlock(&pool_mtx);
    }
    return NULL;
}

// =========================================================================
// MAIN
// =========================================================================
int main(int argc, char *argv[]) {
    int M = (argc > 1) ? atoi(argv[1]) : 45864;
    int K = (argc > 2) ? atoi(argv[2]) : 4;
    int N = (argc > 3) ? atoi(argv[3]) : 32;

    int16_t *Ar = aligned_alloc(32, M*K*2), *Ai = aligned_alloc(32, M*K*2), *Br = aligned_alloc(32, K*N*2), *Bi = aligned_alloc(32, K*N*2);
    int16_t *Cr = aligned_alloc(32, M*N*2), *Ci = aligned_alloc(32, M*N*2);
    
    for (int i = 0; i < num_threads; i++) pthread_create(&pool_threads[i], NULL, worker, &pool_params[i]);

    int Mc_list[] = {16, 32, 64, 128, 140, 168, 180, 256, 280, 512, 1024, M};
    
    printf("=== BENCHMARK Multi-Thread (%d threads) ===\n", num_threads);
    printf("%-8s | %-10s | %-10s | %-10s\n", "Mc", "Min(ms)", "Max(ms)", "Avg(ms)");
    printf("------------------------------------------------------------\n");

    double best_avg = 1e18;
    int best_Mc = 0;

    for (int i = 0; i < sizeof(Mc_list)/sizeof(int); i++) {
        int Mc = Mc_list[i];
        int row_per_th = ((M/num_threads + Mc - 1)/Mc) * Mc;
        for (int t = 0; t < num_threads; t++) {
            pool_params[t] = (ThreadParam){Ar, Ai, Br, Bi, Cr, Ci, M, K, N, Mc, t*row_per_th, MIN((t+1)*row_per_th, M)};
        }
        
        double total_time = 0, min_t = 1e18, max_t = 0;
        int iter = 100;
        for (int n = 0; n < iter; n++) {
            pthread_mutex_lock(&pool_mtx); 
            work_id++; tasks_running = num_threads; pthread_cond_broadcast(&cv_start);
            double t0 = get_time();
            while (tasks_running > 0) pthread_cond_wait(&cv_done, &pool_mtx);
            double t = (get_time() - t0) * 1000.0;
            pthread_mutex_unlock(&pool_mtx);
            total_time += t;
            if (t < min_t) min_t = t;
            if (t > max_t) max_t = t;
        }
        double avg_t = total_time / iter;
        printf("%-8d | %-10.4f | %-10.4f | %-10.4f\n", Mc, min_t, max_t, avg_t);
        
        if (avg_t < best_avg) { best_avg = avg_t; best_Mc = Mc; }
    }

    printf("------------------------------------------------------------\n");
    printf(">>> KẾT QUẢ TỐI ƯU (Avg nhỏ nhất): Mc=%d | Avg=%.4f ms\n", best_Mc, best_avg);

    shutdown_pool = 1; pthread_mutex_lock(&pool_mtx); pthread_cond_broadcast(&cv_start); pthread_mutex_unlock(&pool_mtx);
    for (int i = 0; i < num_threads; i++) pthread_join(pool_threads[i], NULL);
    free(Ar); free(Ai); free(Br); free(Bi); free(Cr); free(Ci);
    return 0;
}