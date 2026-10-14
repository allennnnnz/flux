// hxb CPU-backend "DMA engine" and counter wait, called through ctypes (the GIL is released during the call).
// Persistent worker threads pinned to the DMA cores poll a job sequence number (spin ~1 ms after a copy, then every
// ~10 us), so a copy costs a few us of dispatch instead of the ~0.2 ms a Python thread pool needed (futex wake-ups;
// dry run 2026-10-09).
// Build: gcc -O2 -shared -fPIC -pthread -o libhxbdma.so hxbdma.c
#define _GNU_SOURCE
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <sys/prctl.h>

#define MAXW 64
typedef struct {
    _Atomic uint64_t seq;      // job generation, bumped by the submitter
    _Atomic uint64_t done[MAXW];
    char *dst, *src;
    size_t n, step;
    int nw;
    _Atomic int stop;
    pthread_t th[MAXW];
    int cpu[MAXW];
} pool_t;

static pool_t P;

static void *worker(void *arg) {
    int w = (int)(intptr_t)arg;
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(P.cpu[w], &cs);
    pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    prctl(PR_SET_TIMERSLACK, 1000UL, 0, 0, 0);  // 10 us sleeps really ~10 us
    uint64_t seen = 0;
    long idle = 0;
    while (!atomic_load(&P.stop)) {
        uint64_t s = atomic_load_explicit(&P.seq, memory_order_acquire);
        if (s == seen) {
            // spin ~1 ms after the last copy (back-to-back chunks find the engine hot), then poll every ~10 us so
            // an idle engine does not eat the socket's power budget (spinning cut the compute cores' clock)
            if (++idle < 20000) { _mm_pause(); continue; }
            struct timespec d = {0, 10000};
            nanosleep(&d, 0);
            continue;
        }
        idle = 0;
        seen = s;
        size_t off = (size_t)w * P.step;
        if (off < P.n) {
            size_t len = P.n - off < P.step ? P.n - off : P.step;
            memmove(P.dst + off, P.src + off, len);
        }
        // memmove may use non-temporal stores; they are weakly ordered and only an sfence ON THIS THREAD orders
        // them before the release store below (an sfence in the submitter is not enough: T1 / T4 caught stale
        // reads, 2026-10-09)
        _mm_sfence();
        atomic_store_explicit(&P.done[w], s, memory_order_release);
    }
    return 0;
}

int hxb_dma_start(int nw, const int *cpus) {
    if (nw < 1 || nw > MAXW) return -1;
    P.nw = nw;
    atomic_store(&P.seq, 0);
    atomic_store(&P.stop, 0);
    for (int i = 0; i < nw; i++) { P.cpu[i] = cpus[i]; atomic_store(&P.done[i], 0); }
    for (int i = 0; i < nw; i++)
        if (pthread_create(&P.th[i], 0, worker, (void *)(intptr_t)i)) return -2;
    return 0;
}

// One copy at a time (callers serialise; the backend holds a lock). Returns after every worker finished.
void hxb_dma_copy(void *dst, const void *src, size_t n) {
    size_t step = (n + P.nw - 1) / P.nw;
    step = (step + 63) & ~(size_t)63;
    P.dst = (char *)dst; P.src = (char *)src; P.n = n; P.step = step;
    uint64_t s = atomic_load(&P.seq) + 1;
    atomic_store_explicit(&P.seq, s, memory_order_release);
    for (int i = 0; i < P.nw; i++)
        while (atomic_load_explicit(&P.done[i], memory_order_acquire) != s) _mm_pause();
}

void hxb_dma_stop(void) {
    atomic_store(&P.stop, 1);
    for (int i = 0; i < P.nw; i++) pthread_join(P.th[i], 0);
}

// Wait until *addr >= v (spin, then 2 us sleeps). Returns 0, 1 if *abort != 0, 2 on timeout.
int hxb_wait_geq(volatile int64_t *addr, int64_t v, volatile int64_t *abort_flag, double timeout_s) {
    struct timespec t0, t;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long i = 0;; i++) {
        if (__atomic_load_n(addr, __ATOMIC_ACQUIRE) >= v) return 0;
        if (*abort_flag) return 1;
        if (i < 20000) { _mm_pause(); continue; }
        struct timespec d = {0, 2000};
        nanosleep(&d, 0);
        if ((i & 255) == 0) {
            clock_gettime(CLOCK_MONOTONIC, &t);
            if ((t.tv_sec - t0.tv_sec) + 1e-9 * (t.tv_nsec - t0.tv_nsec) > timeout_s) return 2;
        }
    }
}

// Release store of a counter (monotonic: never lowers it).
void hxb_store_max(volatile int64_t *addr, int64_t v) {
    if (__atomic_load_n(addr, __ATOMIC_ACQUIRE) < v) __atomic_store_n(addr, v, __ATOMIC_RELEASE);
}

// ---- compute kernels of the CPU "device" (OpenMP over rows) ----------------------------------------------------
#include <math.h>

static inline float bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static inline uint16_t f2bf(float f) {  // round to nearest even, as torch's float -> bfloat16 cast
    uint32_t u; memcpy(&u, &f, 4);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// y = w * bf16(x * 1/sqrt(mean(x^2) + eps)), x / w / y bf16 (HF Qwen2RMSNorm order). Replaces 7 torch ops per block:
// 64 rows took 0.54 ms and 8192 rows 15.5 ms (5.4 GB/s) that way (diag_cpu_rmsnorm_v1, 2026-10-14).
void hxb_rmsnorm_bf16(const uint16_t *x, const uint16_t *w, uint16_t *y, long rows, long H, float eps, int nt) {
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (long r = 0; r < rows; r++) {
        const uint16_t *xr = x + r * H;
        uint16_t *yr = y + r * H;
        float acc = 0.f;
        for (long i = 0; i < H; i++) { float v = bf2f(xr[i]); acc += v * v; }
        float inv = 1.0f / sqrtf(acc / (float)H + eps);
        for (long i = 0; i < H; i++) yr[i] = f2bf(bf2f(f2bf(bf2f(xr[i]) * inv)) * bf2f(w[i]));
    }
}

// out[i] = bf16(sum_k fp32(in_k[i])) over k = 0..n_in-1 (host-side all-reduce of the C configuration).
// Summation order: k = 0, 1, ..., n_in-1, in fp32 (NCCL's ring order differs; results may differ by an ulp).
void hxb_sum_bf16(const uint16_t *const *in, int n_in, uint16_t *out, long n, int nt) {
    #pragma omp parallel for num_threads(nt) schedule(static)
    for (long i = 0; i < n; i++) {
        float acc = 0.f;
        for (int k = 0; k < n_in; k++) acc += bf2f(in[k][i]);
        out[i] = f2bf(acc);
    }
}
