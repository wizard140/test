#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

#define MICROSECONDS_PER_SECOND 1000000
#define MICROSECONDS_PER_NANOSECOND 1000

int64_t difftimespec_us(const struct timespec after, const struct timespec before)
{
  return ((int64_t) after.tv_sec - (int64_t) before.tv_sec) * (int64_t) MICROSECONDS_PER_SECOND
       + ((int64_t) after.tv_nsec - (int64_t) before.tv_nsec) / MICROSECONDS_PER_NANOSECOND;
}

void init(float *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = ((float) rand() / (float) RAND_MAX) * 100.0f;
  }
}

void cpuScaleAdd(float *x, float *y, float *z, float c, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    z[i] = c * x[i] + y[i];
  }
}

__global__
void gpuScaleAddAdjacent(float *x, float *y, float *z, float c, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = gridDim.x * blockDim.x;

  for (int i = idx; i < N; i += stride)
  {
    z[i] = c * x[i] + y[i];
  }
}

__global__
void gpuScaleAddBlocks(float *x, float *y, float *z, float c, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_threads = gridDim.x * blockDim.x;
  int chunk_size = (N + total_threads - 1) / total_threads;

  int start = idx * chunk_size;
  int end = start + chunk_size;

  if (end > N)
  {
    end = N;
  }

  for (int i = start; i < end; ++i)
  {
    z[i] = c * x[i] + y[i];
  }
}

int checkSame(float *a, float *b, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (fabsf(a[i] - b[i]) > 1e-4f)
    {
      printf("Mismatch at index %d: %f vs %f\n", i, a[i], b[i]);
      return 0;
    }
  }
  return 1;
}

int64_t timeCpu(float *x, float *y, float *z, float c, int N)
{
  struct timespec start;
  struct timespec end;

  clock_gettime(CLOCK_MONOTONIC, &start);
  cpuScaleAdd(x, y, z, c, N);
  clock_gettime(CLOCK_MONOTONIC, &end);

  return difftimespec_us(end, start);
}

int64_t timeGpuAdjacent(float *hx, float *hy, float *hz,
                        float *dx, float *dy, float *dz,
                        float c, int N,
                        int number_of_blocks, int threads_per_block)
{
  struct timespec start;
  struct timespec end;
  size_t size = (size_t) N * sizeof(float);

  clock_gettime(CLOCK_MONOTONIC, &start);

  cudaMemcpy(dx, hx, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy, size, cudaMemcpyHostToDevice);
  gpuScaleAddAdjacent<<<number_of_blocks, threads_per_block>>>(dx, dy, dz, c, N);
  cudaDeviceSynchronize();
  cudaMemcpy(hz, dz, size, cudaMemcpyDeviceToHost);

  clock_gettime(CLOCK_MONOTONIC, &end);

  return difftimespec_us(end, start);
}

int64_t timeGpuBlocks(float *hx, float *hy, float *hz,
                      float *dx, float *dy, float *dz,
                      float c, int N,
                      int number_of_blocks, int threads_per_block)
{
  struct timespec start;
  struct timespec end;
  size_t size = (size_t) N * sizeof(float);

  clock_gettime(CLOCK_MONOTONIC, &start);

  cudaMemcpy(dx, hx, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy, size, cudaMemcpyHostToDevice);
  gpuScaleAddBlocks<<<number_of_blocks, threads_per_block>>>(dx, dy, dz, c, N);
  cudaDeviceSynchronize();
  cudaMemcpy(hz, dz, size, cudaMemcpyDeviceToHost);

  clock_gettime(CLOCK_MONOTONIC, &end);

  return difftimespec_us(end, start);
}

void runOneTest(FILE *fp, int N, float c, int repeats)
{
  float *hx;
  float *hy;
  float *hz_cpu;
  float *hz_adjacent;
  float *hz_blocks;

  float *dx;
  float *dy;
  float *dz;

  size_t size = (size_t) N * sizeof(float);
  int threads_per_block = 512;
  int number_of_blocks = 80;

  hx = (float *) malloc(size);
  hy = (float *) malloc(size);
  hz_cpu = (float *) malloc(size);
  hz_adjacent = (float *) malloc(size);
  hz_blocks = (float *) malloc(size);

  cudaMalloc((void **) &dx, size);
  cudaMalloc((void **) &dy, size);
  cudaMalloc((void **) &dz, size);

  init(hx, N);
  init(hy, N);

  cpuScaleAdd(hx, hy, hz_cpu, c, N);

  cudaMemcpy(dx, hx, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy, size, cudaMemcpyHostToDevice);
  gpuScaleAddAdjacent<<<number_of_blocks, threads_per_block>>>(dx, dy, dz, c, N);
  cudaDeviceSynchronize();
  cudaMemcpy(hz_adjacent, dz, size, cudaMemcpyDeviceToHost);

  cudaMemcpy(dx, hx, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy, size, cudaMemcpyHostToDevice);
  gpuScaleAddBlocks<<<number_of_blocks, threads_per_block>>>(dx, dy, dz, c, N);
  cudaDeviceSynchronize();
  cudaMemcpy(hz_blocks, dz, size, cudaMemcpyDeviceToHost);

  if (!checkSame(hz_cpu, hz_adjacent, N))
  {
    printf("Adjacent kernel failed verification for N = %d\n", N);
    exit(1);
  }

  if (!checkSame(hz_cpu, hz_blocks, N))
  {
    printf("Block kernel failed verification for N = %d\n", N);
    exit(1);
  }

  int trial;
  int64_t cpu_total = 0;
  int64_t adjacent_total = 0;
  int64_t blocks_total = 0;

  for (trial = 1; trial <= repeats; ++trial)
  {
    cpu_total += timeCpu(hx, hy, hz_cpu, c, N);
    adjacent_total += timeGpuAdjacent(hx, hy, hz_adjacent, dx, dy, dz, c, N, number_of_blocks, threads_per_block);
    blocks_total += timeGpuBlocks(hx, hy, hz_blocks, dx, dy, dz, c, N, number_of_blocks, threads_per_block);
  }

  double cpu_avg = (double) cpu_total / (double) repeats;
  double adjacent_avg = (double) adjacent_total / (double) repeats;
  double blocks_avg = (double) blocks_total / (double) repeats;
  double adjacent_speedup = cpu_avg / adjacent_avg;
  double blocks_speedup = cpu_avg / blocks_avg;

  printf("N=%10d  trials=%2d  cpu=%10.2f us  adjacent=%10.2f us  blocks=%10.2f us\n",
         N, repeats, cpu_avg, adjacent_avg, blocks_avg);

  fprintf(fp, "%d,%d,%d,%d,%.2f,%.2f,%.2f,%.4f,%.4f\n",
          N, repeats, number_of_blocks, threads_per_block,
          cpu_avg, adjacent_avg, blocks_avg,
          adjacent_speedup, blocks_speedup);

  cudaFree(dx);
  cudaFree(dy);
  cudaFree(dz);

  free(hx);
  free(hy);
  free(hz_cpu);
  free(hz_adjacent);
  free(hz_blocks);
}

int main(int argc, char **argv)
{
  int repeats = 11;
  float c = 2.5f;

  if (argc >= 2)
  {
    repeats = atoi(argv[1]);
    if (repeats < 1)
    {
      repeats = 11;
    }
  }

  srand(42);

  FILE *fp = fopen("results.csv", "w");
  if (fp == NULL)
  {
    printf("Could not open results.csv\n");
    return 1;
  }

  fprintf(fp, "N,trials,blocks,threads_per_block,cpu_us,adjacent_us,blocks_us,adjacent_speedup,blocks_speedup\n");

  int sizes[] = {
    1 << 10,
    1 << 12,
    1 << 14,
    1 << 16,
    1 << 18,
    1 << 20,
    1 << 22,
    1 << 24,
    1 << 26
  };

  int count = sizeof(sizes) / sizeof(sizes[0]);

  printf("Running %d timed trials per version for each input size.\n\n", repeats);

  for (int i = 0; i < count; ++i)
  {
    runOneTest(fp, sizes[i], c, repeats);
  }

  fclose(fp);

  printf("\nSaved data to results.csv\n");
  return 0;
}
