#include <cuda.h>
#include <cuda_runtime.h>
#include <numa.h>
#include <numaif.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                                       \
  do {                                                                                         \
    cudaError_t err__ = (call);                                                                \
    if (err__ != cudaSuccess) {                                                                \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
                   cudaGetErrorString(err__));                                                 \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

#define CHECK_DRV(call)                                                                        \
  do {                                                                                         \
    CUresult err__ = (call);                                                                   \
    if (err__ != CUDA_SUCCESS) {                                                               \
      const char* msg__ = nullptr;                                                             \
      cuGetErrorString(err__, &msg__);                                                         \
      std::fprintf(stderr, "CUDA driver error %s:%d: %s\n", __FILE__, __LINE__,              \
                   msg__ ? msg__ : "unknown");                                               \
      std::exit(1);                                                                            \
    }                                                                                          \
  } while (0)

struct Args {
  std::vector<int> devices{0, 1};
  int size_mib = 1024;
  int chunk_mib = 16;
  int warmup = 5;
  int repeats = 20;
  bool ring = false;
};

struct PairCtx {
  int src;
  int dst;
  int numa_node;
  size_t total_bytes;
  size_t chunk_bytes;
  int chunks;
  void* host[2]{};
  unsigned int* flag_host[2]{};
  CUdeviceptr flag_src_dev[2]{};
  CUdeviceptr flag_dst_dev[2]{};
  void* src_dev{};
  void* dst_dev{};
  cudaStream_t src_stream{};
  cudaStream_t dst_stream{};
};

static int gpu_numa_node(int dev) { return dev < 4 ? 0 : 1; }

static std::vector<int> parse_devices(const char* text) {
  std::vector<int> out;
  std::string s(text);
  size_t start = 0;
  while (start < s.size()) {
    size_t comma = s.find(',', start);
    std::string token = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!token.empty()) out.push_back(std::stoi(token));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return out;
}

static Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key(argv[i]);
    auto need_value = [&]() -> const char* {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", key.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (key == "--devices") {
      args.devices = parse_devices(need_value());
    } else if (key == "--size-mib") {
      args.size_mib = std::atoi(need_value());
    } else if (key == "--chunk-mib") {
      args.chunk_mib = std::atoi(need_value());
    } else if (key == "--warmup") {
      args.warmup = std::atoi(need_value());
    } else if (key == "--repeats") {
      args.repeats = std::atoi(need_value());
    } else if (key == "--ring") {
      args.ring = true;
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", key.c_str());
      std::exit(2);
    }
  }
  if (args.devices.size() < 2) {
    std::fprintf(stderr, "need at least two devices\n");
    std::exit(2);
  }
  if (args.size_mib < 1 || args.chunk_mib < 1) {
    std::fprintf(stderr, "size and chunk must be positive\n");
    std::exit(2);
  }
  return args;
}

static void* alloc_pinned_on_node(size_t bytes, int node) {
  void* ptr = numa_alloc_onnode(bytes, node);
  if (ptr == nullptr) {
    std::fprintf(stderr, "numa_alloc_onnode failed\n");
    std::exit(1);
  }
  std::memset(ptr, 0, bytes);
  CHECK_CUDA(cudaHostRegister(ptr, bytes, cudaHostRegisterMapped));
  return ptr;
}

static PairCtx make_pair(int src, int dst, size_t total_bytes, size_t chunk_bytes) {
  PairCtx ctx;
  ctx.src = src;
  ctx.dst = dst;
  ctx.numa_node = gpu_numa_node(src);
  ctx.total_bytes = total_bytes;
  ctx.chunk_bytes = chunk_bytes;
  ctx.chunks = static_cast<int>((total_bytes + chunk_bytes - 1) / chunk_bytes);

  CHECK_CUDA(cudaSetDevice(src));
  CHECK_CUDA(cudaMalloc(&ctx.src_dev, total_bytes));
  CHECK_CUDA(cudaStreamCreateWithFlags(&ctx.src_stream, cudaStreamNonBlocking));
  CHECK_CUDA(cudaSetDevice(dst));
  CHECK_CUDA(cudaMalloc(&ctx.dst_dev, total_bytes));
  CHECK_CUDA(cudaStreamCreateWithFlags(&ctx.dst_stream, cudaStreamNonBlocking));

  for (int i = 0; i < 2; ++i) {
    ctx.host[i] = alloc_pinned_on_node(chunk_bytes, ctx.numa_node);
    ctx.flag_host[i] = static_cast<unsigned int*>(alloc_pinned_on_node(sizeof(unsigned int), ctx.numa_node));
    *ctx.flag_host[i] = 0;
    void* mapped = nullptr;
    CHECK_CUDA(cudaSetDevice(src));
    CHECK_CUDA(cudaHostGetDevicePointer(&mapped, ctx.flag_host[i], 0));
    ctx.flag_src_dev[i] = reinterpret_cast<CUdeviceptr>(mapped);
    CHECK_CUDA(cudaSetDevice(dst));
    CHECK_CUDA(cudaHostGetDevicePointer(&mapped, ctx.flag_host[i], 0));
    ctx.flag_dst_dev[i] = reinterpret_cast<CUdeviceptr>(mapped);
  }
  return ctx;
}

static void destroy_pair(PairCtx& ctx) {
  CHECK_CUDA(cudaSetDevice(ctx.src));
  CHECK_CUDA(cudaStreamDestroy(ctx.src_stream));
  CHECK_CUDA(cudaFree(ctx.src_dev));
  CHECK_CUDA(cudaSetDevice(ctx.dst));
  CHECK_CUDA(cudaStreamDestroy(ctx.dst_stream));
  CHECK_CUDA(cudaFree(ctx.dst_dev));
  for (int i = 0; i < 2; ++i) {
    CHECK_CUDA(cudaHostUnregister(ctx.host[i]));
    numa_free(ctx.host[i], ctx.chunk_bytes);
    CHECK_CUDA(cudaHostUnregister(ctx.flag_host[i]));
    numa_free(ctx.flag_host[i], sizeof(unsigned int));
  }
}

static void enqueue_pair(PairCtx& ctx, unsigned int base_seq) {
  for (int c = 0; c < ctx.chunks; ++c) {
    int b = c & 1;
    size_t off = static_cast<size_t>(c) * ctx.chunk_bytes;
    size_t bytes = std::min(ctx.chunk_bytes, ctx.total_bytes - off);
    unsigned int seq = base_seq + static_cast<unsigned int>(c + 1);

    CHECK_CUDA(cudaSetDevice(ctx.src));
    CHECK_CUDA(cudaMemcpyAsync(ctx.host[b], static_cast<char*>(ctx.src_dev) + off, bytes,
                               cudaMemcpyDeviceToHost, ctx.src_stream));
    CHECK_DRV(cuStreamWriteValue32(reinterpret_cast<CUstream>(ctx.src_stream), ctx.flag_src_dev[b], seq, 0));

    CHECK_CUDA(cudaSetDevice(ctx.dst));
    CHECK_DRV(cuStreamWaitValue32(reinterpret_cast<CUstream>(ctx.dst_stream), ctx.flag_dst_dev[b], seq, CU_STREAM_WAIT_VALUE_GEQ));
    CHECK_CUDA(cudaMemcpyAsync(static_cast<char*>(ctx.dst_dev) + off, ctx.host[b], bytes,
                               cudaMemcpyHostToDevice, ctx.dst_stream));
  }
}

static void synchronize_pair(PairCtx& ctx) {
  CHECK_CUDA(cudaSetDevice(ctx.src));
  CHECK_CUDA(cudaStreamSynchronize(ctx.src_stream));
  CHECK_CUDA(cudaSetDevice(ctx.dst));
  CHECK_CUDA(cudaStreamSynchronize(ctx.dst_stream));
}

static double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  size_t n = values.size();
  if (n % 2) return values[n / 2];
  return (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

static double stdev(const std::vector<double>& values) {
  if (values.size() < 2) return 0.0;
  double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double acc = 0.0;
  for (double v : values) acc += (v - mean) * (v - mean);
  return std::sqrt(acc / (values.size() - 1));
}

int main(int argc, char** argv) {
  Args args = parse_args(argc, argv);
  if (numa_available() < 0) {
    std::fprintf(stderr, "NUMA is unavailable\n");
    return 1;
  }

  size_t total_bytes = static_cast<size_t>(args.size_mib) * 1024 * 1024;
  size_t chunk_bytes = static_cast<size_t>(args.chunk_mib) * 1024 * 1024;
  std::vector<PairCtx> pairs;
  if (args.ring) {
    for (size_t i = 0; i < args.devices.size(); ++i) {
      int src = args.devices[i];
      int dst = args.devices[(i + 1) % args.devices.size()];
      pairs.push_back(make_pair(src, dst, total_bytes, chunk_bytes));
    }
  } else {
    pairs.push_back(make_pair(args.devices[0], args.devices[1], total_bytes, chunk_bytes));
  }

  std::printf("meta,devices=");
  for (size_t i = 0; i < args.devices.size(); ++i) std::printf("%s%d", i ? "|" : "", args.devices[i]);
  std::printf(",ring=%d,size_mib=%d,chunk_mib=%d,warmup=%d,repeats=%d,pairs=%zu\n",
              args.ring ? 1 : 0, args.size_mib, args.chunk_mib, args.warmup, args.repeats, pairs.size());
  std::printf("repeat,wall_seconds,aggregate_payload_GBps,per_pair_payload_GBps\n");

  unsigned int seq_stride = static_cast<unsigned int>((total_bytes + chunk_bytes - 1) / chunk_bytes + 8);
  for (int w = 0; w < args.warmup; ++w) {
    for (auto& pair : pairs) enqueue_pair(pair, static_cast<unsigned int>((w + 1) * seq_stride));
    for (auto& pair : pairs) synchronize_pair(pair);
  }

  std::vector<double> rates;
  for (int r = 0; r < args.repeats; ++r) {
    unsigned int base = static_cast<unsigned int>((args.warmup + r + 1) * seq_stride);
    auto t0 = std::chrono::steady_clock::now();
    for (auto& pair : pairs) enqueue_pair(pair, base);
    for (auto& pair : pairs) synchronize_pair(pair);
    auto t1 = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(t1 - t0).count();
    double payload_gbps = (static_cast<double>(total_bytes) * pairs.size()) / seconds / 1e9;
    rates.push_back(payload_gbps);
    std::printf("%d,%.9f,%.3f,%.3f\n", r, seconds, payload_gbps, payload_gbps / pairs.size());
  }

  std::printf("summary,aggregate_median_GBps=%.3f,aggregate_stdev_GBps=%.3f,per_pair_median_GBps=%.3f\n",
              median(rates), stdev(rates), median(rates) / pairs.size());

  for (auto& pair : pairs) destroy_pair(pair);
  return 0;
}
