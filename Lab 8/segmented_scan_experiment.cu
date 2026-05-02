
#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define DEFAULT_REPEATS 5
#define DEFAULT_MIN_POWER 20
#define DEFAULT_MAX_POWER 26
#define DEFAULT_VERIFY_LIMIT 4194304

#define BLOCK_SIZE 256
#define PATTERN_COUNT 4

typedef struct
{
  const char *name;
  int mode;
  int segment_size;
  float random_probability;
} SegmentPattern;

static SegmentPattern patterns[PATTERN_COUNT] =
{
  {"small_segments_every_8", 0, 8, 0.0f},
  {"medium_segments_every_64", 0, 64, 0.0f},
  {"large_segments_every_1024", 0, 1024, 0.0f},
  {"random_segments_10_percent", 1, 0, 0.10f}
};

static void cudaCheck(cudaError_t status, const char *message)
{
  if (status != cudaSuccess)
  {
    fprintf(stderr, "CUDA error at %s: %s\n", message, cudaGetErrorString(status));
    exit(1);
  }
}

static void *checkedMalloc(size_t bytes, const char *message)
{
  void *ptr = malloc(bytes);

  if (ptr == NULL)
  {
    fprintf(stderr, "Could not allocate %zu bytes for %s\n", bytes, message);
    exit(1);
  }

  return ptr;
}

static double nowMs()
{
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1000000.0;
}

/*
  Repeatable random number generator.

  This keeps the input the same every run, which makes the experiment easier
  to verify and compare.
*/
static unsigned int nextRandom(unsigned int *state)
{
  *state = (*state * 1664525u) + 1013904223u;
  return *state;
}

/*
  Creates values and segment flags.

  values[i] is the number being added.
  flags[i] tells if a new segment starts.

  flags[i] == 1 means restart the scan at this element.
  flags[i] == 0 means continue adding from the previous element.
*/
static void initInput(float *values, int *flags, int n, SegmentPattern pattern)
{
  int i;
  unsigned int state = 12345u;

  for (i = 0; i < n; ++i)
  {
    values[i] = (float)((nextRandom(&state) % 10u) + 1u);

    if (i == 0)
    {
      flags[i] = 1;
    }
    else if (pattern.mode == 0)
    {
      flags[i] = (i % pattern.segment_size == 0) ? 1 : 0;
    }
    else
    {
      unsigned int r = nextRandom(&state) % 1000u;
      flags[i] = (r < (unsigned int)(pattern.random_probability * 1000.0f)) ? 1 : 0;
    }
  }
}

/*
  CPU segmented inclusive scan.

  This is used as the correctness baseline for smaller input sizes.
*/
static void cpuSegmentedInclusiveScan(const float *values,
                                      const int *flags,
                                      float *output,
                                      int n)
{
  int i;
  float running_sum = 0.0f;

  for (i = 0; i < n; ++i)
  {
    if (flags[i] == 1)
    {
      running_sum = values[i];
    }
    else
    {
      running_sum += values[i];
    }

    output[i] = running_sum;
  }
}

static int checkSame(const float *cpu_output, const float *gpu_output, int n)
{
  int i;

  for (i = 0; i < n; ++i)
  {
    if (fabsf(cpu_output[i] - gpu_output[i]) > 1.0e-3f)
    {
      printf("Mismatch at index %d: CPU %.3f GPU %.3f\n",
             i,
             cpu_output[i],
             gpu_output[i]);
      return 0;
    }
  }

  return 1;
}

/*
  This is the key segmented-scan combine operation.

  left_value and left_flag are from the left side.
  right_value and right_flag are from the right side.

  If right_flag is 1, the right side starts a new segment, so the left side
  should not be added.

  If right_flag is 0, the right side is part of the same segment, so the left
  partial sum is added.
*/
__device__ void combinePair(float left_value,
                            int left_flag,
                            float right_value,
                            int right_flag,
                            float *out_value,
                            int *out_flag)
{
  if (right_flag == 1)
  {
    *out_value = right_value;
  }
  else
  {
    *out_value = left_value + right_value;
  }

  *out_flag = left_flag | right_flag;
}

/*
  Copies one pair array into another.

  Brent-Kung only updates some indexes on each pass, so this copy keeps the
  unchanged indexes valid.
*/
__global__ void copyPairKernel(const float *in_values,
                               const int *in_flags,
                               float *out_values,
                               int *out_flags,
                               int n)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int i;

  for (i = tid; i < n; i += stride)
  {
    out_values[i] = in_values[i];
    out_flags[i] = in_flags[i];
  }
}

/*
  One full-array Kogge-Stone step.

  offset goes 1, 2, 4, 8, ...
  Every element reads from i - offset if that index exists.

  This works across many CUDA blocks because each step is a separate kernel
  launch. A kernel launch acts like a global synchronization point.
*/
__global__ void koggeStoneStepKernel(const float *in_values,
                                     const int *in_flags,
                                     float *out_values,
                                     int *out_flags,
                                     int n,
                                     int offset)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int i;

  for (i = tid; i < n; i += stride)
  {
    float my_value = in_values[i];
    int my_flag = in_flags[i];

    if (i >= offset)
    {
      float left_value = in_values[i - offset];
      int left_flag = in_flags[i - offset];

      combinePair(left_value,
                  left_flag,
                  my_value,
                  my_flag,
                  &my_value,
                  &my_flag);
    }

    out_values[i] = my_value;
    out_flags[i] = my_flag;
  }
}

/*
  One full-array Brent-Kung step.

  phase 0 is upsweep.
  phase 1 is downsweep.

*/
__global__ void brentKungStepKernel(const float *in_values,
                                    const int *in_flags,
                                    float *out_values,
                                    int *out_flags,
                                    int n,
                                    int stride_amount,
                                    int phase)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;
  int t;

  for (t = tid; t < n; t += grid_stride)
  {
    if (phase == 0)
    {
      int index = ((t + 1) * stride_amount * 2) - 1;

      if (index < n)
      {
        float new_value;
        int new_flag;

        combinePair(in_values[index - stride_amount],
                    in_flags[index - stride_amount],
                    in_values[index],
                    in_flags[index],
                    &new_value,
                    &new_flag);

        out_values[index] = new_value;
        out_flags[index] = new_flag;
      }
    }
    else
    {
      int index = ((t + 1) * stride_amount * 2) - 1;
      int right_index = index + stride_amount;

      if (right_index < n)
      {
        float new_value;
        int new_flag;

        combinePair(in_values[index],
                    in_flags[index],
                    in_values[right_index],
                    in_flags[right_index],
                    &new_value,
                    &new_flag);

        out_values[right_index] = new_value;
        out_flags[right_index] = new_flag;
      }
    }
  }
}

static int makeBlockCount(int n)
{
  int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  /*
    Keep the grid size reasonable.

    The grid-stride loops still cover the whole array even if the number of
    blocks is capped here.
  */
  if (blocks > 65535)
  {
    blocks = 65535;
  }

  return blocks;
}

static double timeCpu(const float *values,
                      const int *flags,
                      float *output,
                      int n)
{
  double start = nowMs();

  cpuSegmentedInclusiveScan(values, flags, output, n);

  return nowMs() - start;
}

/*
  Multi-block Kogge-Stone segmented scan.

  This uses two GPU buffers:
    A = current data
    B = next data

  After each offset pass, the buffers swap.
*/
static double timeGpuKoggeStone(const float *h_values,
                                const int *h_flags,
                                float *h_output,
                                int n)
{
  float *d_values_a;
  float *d_values_b;
  int *d_flags_a;
  int *d_flags_b;

  size_t value_bytes = (size_t)n * sizeof(float);
  size_t flag_bytes = (size_t)n * sizeof(int);

  int offset;
  int blocks = makeBlockCount(n);

  cudaCheck(cudaMalloc((void **)&d_values_a, value_bytes), "cudaMalloc d_values_a");
  cudaCheck(cudaMalloc((void **)&d_values_b, value_bytes), "cudaMalloc d_values_b");
  cudaCheck(cudaMalloc((void **)&d_flags_a, flag_bytes), "cudaMalloc d_flags_a");
  cudaCheck(cudaMalloc((void **)&d_flags_b, flag_bytes), "cudaMalloc d_flags_b");

  double start = nowMs();

  cudaCheck(cudaMemcpy(d_values_a, h_values, value_bytes, cudaMemcpyHostToDevice),
            "copy Kogge-Stone values to GPU");
  cudaCheck(cudaMemcpy(d_flags_a, h_flags, flag_bytes, cudaMemcpyHostToDevice),
            "copy Kogge-Stone flags to GPU");

  for (offset = 1; offset < n; offset *= 2)
  {
    koggeStoneStepKernel<<<blocks, BLOCK_SIZE>>>(d_values_a,
                                                 d_flags_a,
                                                 d_values_b,
                                                 d_flags_b,
                                                 n,
                                                 offset);

    cudaCheck(cudaGetLastError(), "launch Kogge-Stone step");
    cudaCheck(cudaDeviceSynchronize(), "sync Kogge-Stone step");

    {
      float *temp_values = d_values_a;
      int *temp_flags = d_flags_a;

      d_values_a = d_values_b;
      d_flags_a = d_flags_b;

      d_values_b = temp_values;
      d_flags_b = temp_flags;
    }
  }

  cudaCheck(cudaMemcpy(h_output, d_values_a, value_bytes, cudaMemcpyDeviceToHost),
            "copy Kogge-Stone output to CPU");

  double elapsed = nowMs() - start;

  cudaFree(d_values_a);
  cudaFree(d_values_b);
  cudaFree(d_flags_a);
  cudaFree(d_flags_b);

  return elapsed;
}

/*
  Multi-block Brent-Kung segmented scan.

  This version does:
    upsweep
    downsweep

  It uses a copy before each tree step because not every index updates at every
  Brent-Kung level.
*/
static double timeGpuBrentKung(const float *h_values,
                               const int *h_flags,
                               float *h_output,
                               int n)
{
  float *d_values_a;
  float *d_values_b;
  int *d_flags_a;
  int *d_flags_b;

  size_t value_bytes = (size_t)n * sizeof(float);
  size_t flag_bytes = (size_t)n * sizeof(int);

  int stride_amount;
  int blocks = makeBlockCount(n);

  cudaCheck(cudaMalloc((void **)&d_values_a, value_bytes), "cudaMalloc d_values_a");
  cudaCheck(cudaMalloc((void **)&d_values_b, value_bytes), "cudaMalloc d_values_b");
  cudaCheck(cudaMalloc((void **)&d_flags_a, flag_bytes), "cudaMalloc d_flags_a");
  cudaCheck(cudaMalloc((void **)&d_flags_b, flag_bytes), "cudaMalloc d_flags_b");

  double start = nowMs();

  cudaCheck(cudaMemcpy(d_values_a, h_values, value_bytes, cudaMemcpyHostToDevice),
            "copy Brent-Kung values to GPU");
  cudaCheck(cudaMemcpy(d_flags_a, h_flags, flag_bytes, cudaMemcpyHostToDevice),
            "copy Brent-Kung flags to GPU");

  for (stride_amount = 1; stride_amount < n; stride_amount *= 2)
  {
    copyPairKernel<<<blocks, BLOCK_SIZE>>>(d_values_a,
                                           d_flags_a,
                                           d_values_b,
                                           d_flags_b,
                                           n);

    cudaCheck(cudaGetLastError(), "launch Brent-Kung copy upsweep");
    cudaCheck(cudaDeviceSynchronize(), "sync Brent-Kung copy upsweep");

    brentKungStepKernel<<<blocks, BLOCK_SIZE>>>(d_values_a,
                                                d_flags_a,
                                                d_values_b,
                                                d_flags_b,
                                                n,
                                                stride_amount,
                                                0);

    cudaCheck(cudaGetLastError(), "launch Brent-Kung upsweep");
    cudaCheck(cudaDeviceSynchronize(), "sync Brent-Kung upsweep");

    {
      float *temp_values = d_values_a;
      int *temp_flags = d_flags_a;

      d_values_a = d_values_b;
      d_flags_a = d_flags_b;

      d_values_b = temp_values;
      d_flags_b = temp_flags;
    }
  }

  for (stride_amount = n / 4; stride_amount >= 1; stride_amount /= 2)
  {
    copyPairKernel<<<blocks, BLOCK_SIZE>>>(d_values_a,
                                           d_flags_a,
                                           d_values_b,
                                           d_flags_b,
                                           n);

    cudaCheck(cudaGetLastError(), "launch Brent-Kung copy downsweep");
    cudaCheck(cudaDeviceSynchronize(), "sync Brent-Kung copy downsweep");

    brentKungStepKernel<<<blocks, BLOCK_SIZE>>>(d_values_a,
                                                d_flags_a,
                                                d_values_b,
                                                d_flags_b,
                                                n,
                                                stride_amount,
                                                1);

    cudaCheck(cudaGetLastError(), "launch Brent-Kung downsweep");
    cudaCheck(cudaDeviceSynchronize(), "sync Brent-Kung downsweep");

    {
      float *temp_values = d_values_a;
      int *temp_flags = d_flags_a;

      d_values_a = d_values_b;
      d_flags_a = d_flags_b;

      d_values_b = temp_values;
      d_flags_b = temp_flags;
    }
  }

  cudaCheck(cudaMemcpy(h_output, d_values_a, value_bytes, cudaMemcpyDeviceToHost),
            "copy Brent-Kung output to CPU");

  double elapsed = nowMs() - start;

  cudaFree(d_values_a);
  cudaFree(d_values_b);
  cudaFree(d_flags_a);
  cudaFree(d_flags_b);

  return elapsed;
}

static void runOneCase(FILE *fp,
                       int n,
                       SegmentPattern pattern,
                       int repeats,
                       int verify_limit)
{
  int repeat;
  int verified_bk = -1;
  int verified_ks = -1;

  size_t value_bytes = (size_t)n * sizeof(float);
  size_t flag_bytes = (size_t)n * sizeof(int);

  float *values = (float *)checkedMalloc(value_bytes, "values");
  int *flags = (int *)checkedMalloc(flag_bytes, "flags");

  float *bk_output = (float *)checkedMalloc(value_bytes, "bk_output");
  float *ks_output = (float *)checkedMalloc(value_bytes, "ks_output");
  float *cpu_output = NULL;

  double cpu_total = 0.0;
  double bk_total = 0.0;
  double ks_total = 0.0;

  initInput(values, flags, n, pattern);

  if (n <= verify_limit)
  {
    cpu_output = (float *)checkedMalloc(value_bytes, "cpu_output");
  }

  for (repeat = 0; repeat < repeats; ++repeat)
  {
    double cpu_time = -1.0;
    double bk_time;
    double ks_time;

    if (cpu_output != NULL)
    {
      cpu_time = timeCpu(values, flags, cpu_output, n);
      cpu_total += cpu_time;
    }

    bk_time = timeGpuBrentKung(values, flags, bk_output, n);
    ks_time = timeGpuKoggeStone(values, flags, ks_output, n);

    bk_total += bk_time;
    ks_total += ks_time;

    if (cpu_output != NULL)
    {
      verified_bk = checkSame(cpu_output, bk_output, n);
      verified_ks = checkSame(cpu_output, ks_output, n);
    }

    fprintf(fp,
            "%d,%s,%d,%f,%f,%f,%d,%d\n",
            n,
            pattern.name,
            repeat + 1,
            cpu_time,
            bk_time,
            ks_time,
            verified_bk,
            verified_ks);
  }

  {
    double cpu_avg = (cpu_output != NULL) ? cpu_total / (double)repeats : -1.0;
    double bk_avg = bk_total / (double)repeats;
    double ks_avg = ks_total / (double)repeats;

    printf("%10d  %-28s  CPU %10.4f ms  BK %10.4f ms  KS %10.4f ms  verified %d/%d\n",
           n,
           pattern.name,
           cpu_avg,
           bk_avg,
           ks_avg,
           verified_bk,
           verified_ks);
  }

  free(values);
  free(flags);
  free(bk_output);
  free(ks_output);

  if (cpu_output != NULL)
  {
    free(cpu_output);
  }
}

int main(int argc, char **argv)
{
  int repeats = DEFAULT_REPEATS;
  int min_power = DEFAULT_MIN_POWER;
  int max_power = DEFAULT_MAX_POWER;
  int verify_limit = DEFAULT_VERIFY_LIMIT;

  int power;
  int j;

  FILE *fp;

  cudaDeviceProp prop;
  int device = 0;

  if (argc >= 2)
  {
    repeats = atoi(argv[1]);
  }

  if (argc >= 4)
  {
    min_power = atoi(argv[2]);
    max_power = atoi(argv[3]);
  }

  if (argc >= 5)
  {
    verify_limit = atoi(argv[4]);
  }

  if (repeats <= 0)
  {
    repeats = DEFAULT_REPEATS;
  }

  if (min_power < 1 || max_power < min_power || max_power > 29)
  {
    fprintf(stderr, "Use powers from 1 to 29, with minPower <= maxPower.\n");
    return 1;
  }

  cudaCheck(cudaGetDevice(&device), "cudaGetDevice");
  cudaCheck(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

  fp = fopen("segmented_scan_multiblock_results.csv", "w");

  if (fp == NULL)
  {
    fprintf(stderr, "Could not open segmented_scan_multiblock_results.csv\n");
    return 1;
  }

  fprintf(fp,
          "n,pattern,repeat,cpu_ms,brent_kung_ms,kogge_stone_ms,brent_kung_verified,kogge_stone_verified\n");

  printf("Multi-Block Segmented Inclusive Scan Experiment\n");
  printf("GPU: %s\n", prop.name);
  printf("Repeats per case: %d\n", repeats);
  printf("Testing powers: 2^%d through 2^%d\n", min_power, max_power);
  printf("CPU verification limit: %d elements\n\n", verify_limit);

  for (power = min_power; power <= max_power; ++power)
  {
    int n = 1 << power;

    for (j = 0; j < PATTERN_COUNT; ++j)
    {
      runOneCase(fp, n, patterns[j], repeats, verify_limit);
    }
  }

  fclose(fp);

  printf("\nResults saved to segmented_scan_multiblock_results.csv\n");

  return 0;
}
