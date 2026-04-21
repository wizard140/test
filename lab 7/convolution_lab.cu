#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <chrono>

#define MAX_FILTER_WIDTH 31
#define GLOBAL_THREADS_X 32
#define GLOBAL_THREADS_Y 8
#define TILE_THREADS_X 16
#define TILE_THREADS_Y 16
#define DEFAULT_REPEATS 7
#define FILTER_COUNT 4
#define IMAGE_COUNT 4

#define VERSION_CPU 0
#define VERSION_GLOBAL 1
#define VERSION_TILED 2
#define VERSION_CONSTANT 3
#define VERSION_TILED_CONSTANT 4
#define VERSION_COUNT 5

__constant__ float constant_filter[MAX_FILTER_WIDTH * MAX_FILTER_WIDTH];

typedef struct
{
  const char *name;
  int width;
  int height;
  int pattern;
} ImageSpec;

typedef struct
{
  const char *name;
  int width;
  int height;
  const float *values;
} FilterSpec;

static const float filter_emboss_3x3[] = {
  -2.0f, -1.0f,  0.0f,
  -1.0f,  1.0f,  1.0f,
   0.0f,  1.0f,  2.0f
};

static const float filter_sharpen_3x3[] = {
   0.0f, -1.0f,  0.0f,
  -1.0f,  5.0f, -1.0f,
   0.0f, -1.0f,  0.0f
};

static const float filter_gaussian_5x5[] = {
   1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f, 1.0f / 273.0f,
   4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
   7.0f / 273.0f, 26.0f / 273.0f, 41.0f / 273.0f, 26.0f / 273.0f, 7.0f / 273.0f,
   4.0f / 273.0f, 16.0f / 273.0f, 26.0f / 273.0f, 16.0f / 273.0f, 4.0f / 273.0f,
   1.0f / 273.0f,  4.0f / 273.0f,  7.0f / 273.0f,  4.0f / 273.0f, 1.0f / 273.0f
};

static const float filter_edge_5x5[] = {
   0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
   0.0f, -1.0f, -2.0f, -1.0f,  0.0f,
  -1.0f, -2.0f, 16.0f, -2.0f, -1.0f,
   0.0f, -1.0f, -2.0f, -1.0f,  0.0f,
   0.0f,  0.0f, -1.0f,  0.0f,  0.0f
};

static const ImageSpec images[IMAGE_COUNT] = {
  { "gradient_512",  512,  512, 0 },
  { "checker_1024", 1024, 1024, 1 },
  { "rings_2048",   2048, 2048, 2 },
  { "noise_3072",   3072, 3072, 3 }
};

static const FilterSpec filters[FILTER_COUNT] = {
  { "emboss_3x3",   3, 3, filter_emboss_3x3 },
  { "sharpen_3x3",  3, 3, filter_sharpen_3x3 },
  { "gaussian_5x5", 5, 5, filter_gaussian_5x5 },
  { "edge_5x5",     5, 5, filter_edge_5x5 }
};

static const char *version_names[VERSION_COUNT] = {
  "cpu_single",
  "gpu_global",
  "gpu_tiled",
  "gpu_constant",
  "gpu_tiled_constant"
};

static inline double nowMs(void)
{
  using clock_type = std::chrono::steady_clock;
  return std::chrono::duration<double, std::milli>(clock_type::now().time_since_epoch()).count();
}

static void cudaCheck(cudaError_t status, const char *where)
{
  if (status != cudaSuccess)
  {
    fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(status));
    exit(1);
  }
}

static void *checkedMalloc(size_t size)
{
  void *ptr = malloc(size);

  if (ptr == NULL)
  {
    fprintf(stderr, "malloc failed for %zu bytes\n", size);
    exit(1);
  }

  return ptr;
}

static uint32_t nextRandom(uint32_t *state)
{
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

static void initImage(float *image, int width, int height, int pattern)
{
  int x;
  int y;
  uint32_t state = 0x12345678u;
  float cx = 0.5f * (float) width;
  float cy = 0.5f * (float) height;

  for (y = 0; y < height; ++y)
  {
    for (x = 0; x < width; ++x)
    {
      float value = 0.0f;
      float fx = (float) x / (float) (width > 1 ? width - 1 : 1);
      float fy = (float) y / (float) (height > 1 ? height - 1 : 1);

      if (pattern == 0)
      {
        value = 255.0f * (0.65f * fx + 0.35f * fy);
      }
      else if (pattern == 1)
      {
        int block = 32;
        int a = (x / block) & 1;
        int b = (y / block) & 1;
        value = (a ^ b) ? 230.0f : 25.0f;
      }
      else if (pattern == 2)
      {
        float dx = (float) x - cx;
        float dy = (float) y - cy;
        float r = sqrtf(dx * dx + dy * dy);
        value = 127.5f + 127.5f * sinf(r * 0.03f + fx * 8.0f);
      }
      else
      {
        value = (float) (nextRandom(&state) & 255u);
      }

      image[y * width + x] = value;
    }
  }
}

static void normalizeToBytes(const float *input, unsigned char *output, int count)
{
  int i;
  float min_value = input[0];
  float max_value = input[0];

  for (i = 1; i < count; ++i)
  {
    if (input[i] < min_value)
    {
      min_value = input[i];
    }

    if (input[i] > max_value)
    {
      max_value = input[i];
    }
  }

  if (fabsf(max_value - min_value) < 1.0e-8f)
  {
    for (i = 0; i < count; ++i)
    {
      output[i] = 0;
    }

    return;
  }

  for (i = 0; i < count; ++i)
  {
    float scaled = 255.0f * (input[i] - min_value) / (max_value - min_value);

    if (scaled < 0.0f)
    {
      scaled = 0.0f;
    }

    if (scaled > 255.0f)
    {
      scaled = 255.0f;
    }

    output[i] = (unsigned char) (scaled + 0.5f);
  }
}

static void writePgm(const char *path, const float *image, int width, int height)
{
  FILE *fp = fopen(path, "wb");
  int count = width * height;
  unsigned char *buffer;

  if (fp == NULL)
  {
    fprintf(stderr, "Could not open %s for writing\n", path);
    exit(1);
  }

  buffer = (unsigned char *) checkedMalloc((size_t) count);
  normalizeToBytes(image, buffer, count);

  fprintf(fp, "P5\n%d %d\n255\n", width, height);
  fwrite(buffer, 1, (size_t) count, fp);
  fclose(fp);
  free(buffer);
}

static int checkSame(const float *a, const float *b, int count)
{
  int i;

  for (i = 0; i < count; ++i)
  {
    float diff = fabsf(a[i] - b[i]);
    float scale = fmaxf(1.0f, fabsf(a[i]));

    if (diff > 1.0e-3f * scale)
    {
      return 0;
    }
  }

  return 1;
}

static void cpuConvolution(const float *input,
                           float *output,
                           int width,
                           int height,
                           const float *filter,
                           int filter_width,
                           int filter_height)
{
  int x;
  int y;
  int radius_x = filter_width / 2;
  int radius_y = filter_height / 2;

  for (y = 0; y < height; ++y)
  {
    for (x = 0; x < width; ++x)
    {
      float sum = 0.0f;
      int fx;
      int fy;

      for (fy = 0; fy < filter_height; ++fy)
      {
        int iy = y + fy - radius_y;

        for (fx = 0; fx < filter_width; ++fx)
        {
          int ix = x + fx - radius_x;

          if (ix >= 0 && ix < width && iy >= 0 && iy < height)
          {
            sum += input[iy * width + ix] * filter[fy * filter_width + fx];
          }
        }
      }

      output[y * width + x] = sum;
    }
  }
}

__global__
static void gpuConvolutionGlobal(const float *input,
                                 float *output,
                                 int width,
                                 int height,
                                 const float *filter,
                                 int filter_width,
                                 int filter_height)
{
  int x_stride = blockDim.x * gridDim.x;
  int y_stride = blockDim.y * gridDim.y;
  int radius_x = filter_width / 2;
  int radius_y = filter_height / 2;
  int y;

  for (y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += y_stride)
  {
    int x;

    for (x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += x_stride)
    {
      float sum = 0.0f;
      int fx;
      int fy;

      for (fy = 0; fy < filter_height; ++fy)
      {
        int iy = y + fy - radius_y;

        for (fx = 0; fx < filter_width; ++fx)
        {
          int ix = x + fx - radius_x;

          if (ix >= 0 && ix < width && iy >= 0 && iy < height)
          {
            sum += input[iy * width + ix] * filter[fy * filter_width + fx];
          }
        }
      }

      output[y * width + x] = sum;
    }
  }
}

__global__
static void gpuConvolutionConstant(const float *input,
                                   float *output,
                                   int width,
                                   int height,
                                   int filter_width,
                                   int filter_height)
{
  int x_stride = blockDim.x * gridDim.x;
  int y_stride = blockDim.y * gridDim.y;
  int radius_x = filter_width / 2;
  int radius_y = filter_height / 2;
  int y;

  for (y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += y_stride)
  {
    int x;

    for (x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += x_stride)
    {
      float sum = 0.0f;
      int fx;
      int fy;

      for (fy = 0; fy < filter_height; ++fy)
      {
        int iy = y + fy - radius_y;

        for (fx = 0; fx < filter_width; ++fx)
        {
          int ix = x + fx - radius_x;

          if (ix >= 0 && ix < width && iy >= 0 && iy < height)
          {
            sum += input[iy * width + ix] * constant_filter[fy * filter_width + fx];
          }
        }
      }

      output[y * width + x] = sum;
    }
  }
}

__global__
static void gpuConvolutionTiled(const float *input,
                                float *output,
                                int width,
                                int height,
                                const float *filter,
                                int filter_width,
                                int filter_height)
{
  int radius_x = filter_width / 2;
  int radius_y = filter_height / 2;
  int shared_width = blockDim.x + 2 * radius_x;
  int shared_height = blockDim.y + 2 * radius_y;
  int base_x = blockIdx.x * blockDim.x;
  int base_y = blockIdx.y * blockDim.y;
  int x = base_x + threadIdx.x;
  int y = base_y + threadIdx.y;
  int tile_x;
  int tile_y;
  extern __shared__ float tile[];

  for (tile_y = threadIdx.y; tile_y < shared_height; tile_y += blockDim.y)
  {
    for (tile_x = threadIdx.x; tile_x < shared_width; tile_x += blockDim.x)
    {
      int global_x = base_x + tile_x - radius_x;
      int global_y = base_y + tile_y - radius_y;

      if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height)
      {
        tile[tile_y * shared_width + tile_x] = input[global_y * width + global_x];
      }
      else
      {
        tile[tile_y * shared_width + tile_x] = 0.0f;
      }
    }
  }

  __syncthreads();

  if (x < width && y < height)
  {
    float sum = 0.0f;
    int fx;
    int fy;

    for (fy = 0; fy < filter_height; ++fy)
    {
      for (fx = 0; fx < filter_width; ++fx)
      {
        float pixel = tile[(threadIdx.y + fy) * shared_width + (threadIdx.x + fx)];
        sum += pixel * filter[fy * filter_width + fx];
      }
    }

    output[y * width + x] = sum;
  }
}

__global__
static void gpuConvolutionTiledConstant(const float *input,
                                        float *output,
                                        int width,
                                        int height,
                                        int filter_width,
                                        int filter_height)
{
  int radius_x = filter_width / 2;
  int radius_y = filter_height / 2;
  int shared_width = blockDim.x + 2 * radius_x;
  int shared_height = blockDim.y + 2 * radius_y;
  int base_x = blockIdx.x * blockDim.x;
  int base_y = blockIdx.y * blockDim.y;
  int x = base_x + threadIdx.x;
  int y = base_y + threadIdx.y;
  int tile_x;
  int tile_y;
  extern __shared__ float tile[];

  for (tile_y = threadIdx.y; tile_y < shared_height; tile_y += blockDim.y)
  {
    for (tile_x = threadIdx.x; tile_x < shared_width; tile_x += blockDim.x)
    {
      int global_x = base_x + tile_x - radius_x;
      int global_y = base_y + tile_y - radius_y;

      if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height)
      {
        tile[tile_y * shared_width + tile_x] = input[global_y * width + global_x];
      }
      else
      {
        tile[tile_y * shared_width + tile_x] = 0.0f;
      }
    }
  }

  __syncthreads();

  if (x < width && y < height)
  {
    float sum = 0.0f;
    int fx;
    int fy;

    for (fy = 0; fy < filter_height; ++fy)
    {
      for (fx = 0; fx < filter_width; ++fx)
      {
        float pixel = tile[(threadIdx.y + fy) * shared_width + (threadIdx.x + fx)];
        sum += pixel * constant_filter[fy * filter_width + fx];
      }
    }

    output[y * width + x] = sum;
  }
}

static dim3 makeGlobalThreads(void)
{
  return dim3(GLOBAL_THREADS_X, GLOBAL_THREADS_Y);
}

static dim3 makeGlobalBlocks(int width, int height)
{
  dim3 threads = makeGlobalThreads();
  return dim3((unsigned int) ((width + threads.x - 1) / threads.x),
              (unsigned int) ((height + threads.y - 1) / threads.y));
}

static dim3 makeTileThreads(void)
{
  return dim3(TILE_THREADS_X, TILE_THREADS_Y);
}

static dim3 makeTileBlocks(int width, int height)
{
  dim3 threads = makeTileThreads();
  return dim3((unsigned int) ((width + threads.x - 1) / threads.x),
              (unsigned int) ((height + threads.y - 1) / threads.y));
}

static size_t sharedBytesForTile(int filter_width, int filter_height)
{
  int shared_width = TILE_THREADS_X + filter_width - 1;
  int shared_height = TILE_THREADS_Y + filter_height - 1;
  return (size_t) shared_width * (size_t) shared_height * sizeof(float);
}

static double timeCpu(const float *input,
                      float *output,
                      int width,
                      int height,
                      const float *filter,
                      int filter_width,
                      int filter_height)
{
  double start = nowMs();

  cpuConvolution(input, output, width, height, filter, filter_width, filter_height);

  return nowMs() - start;
}

static double timeGpuGlobal(const float *h_input,
                            float *h_output,
                            float *d_input,
                            float *d_output,
                            const float *d_filter,
                            size_t image_bytes,
                            int width,
                            int height,
                            int filter_width,
                            int filter_height)
{
  dim3 threads = makeGlobalThreads();
  dim3 blocks = makeGlobalBlocks(width, height);
  double start = nowMs();

  cudaCheck(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice), "cudaMemcpy global input");
  gpuConvolutionGlobal<<<blocks, threads>>>(d_input, d_output, width, height, d_filter, filter_width, filter_height);
  cudaCheck(cudaPeekAtLastError(), "gpuConvolutionGlobal launch");
  cudaCheck(cudaDeviceSynchronize(), "gpuConvolutionGlobal sync");
  cudaCheck(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy global output");

  return nowMs() - start;
}

static double timeGpuTiled(const float *h_input,
                           float *h_output,
                           float *d_input,
                           float *d_output,
                           const float *d_filter,
                           size_t image_bytes,
                           int width,
                           int height,
                           int filter_width,
                           int filter_height)
{
  dim3 threads = makeTileThreads();
  dim3 blocks = makeTileBlocks(width, height);
  size_t shared_bytes = sharedBytesForTile(filter_width, filter_height);
  double start = nowMs();

  cudaCheck(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice), "cudaMemcpy tiled input");
  gpuConvolutionTiled<<<blocks, threads, shared_bytes>>>(d_input,
                                                         d_output,
                                                         width,
                                                         height,
                                                         d_filter,
                                                         filter_width,
                                                         filter_height);
  cudaCheck(cudaPeekAtLastError(), "gpuConvolutionTiled launch");
  cudaCheck(cudaDeviceSynchronize(), "gpuConvolutionTiled sync");
  cudaCheck(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy tiled output");

  return nowMs() - start;
}

static double timeGpuConstant(const float *h_input,
                              float *h_output,
                              float *d_input,
                              float *d_output,
                              size_t image_bytes,
                              int width,
                              int height,
                              int filter_width,
                              int filter_height)
{
  dim3 threads = makeGlobalThreads();
  dim3 blocks = makeGlobalBlocks(width, height);
  double start = nowMs();

  cudaCheck(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice), "cudaMemcpy constant input");
  gpuConvolutionConstant<<<blocks, threads>>>(d_input, d_output, width, height, filter_width, filter_height);
  cudaCheck(cudaPeekAtLastError(), "gpuConvolutionConstant launch");
  cudaCheck(cudaDeviceSynchronize(), "gpuConvolutionConstant sync");
  cudaCheck(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy constant output");

  return nowMs() - start;
}

static double timeGpuTiledConstant(const float *h_input,
                                   float *h_output,
                                   float *d_input,
                                   float *d_output,
                                   size_t image_bytes,
                                   int width,
                                   int height,
                                   int filter_width,
                                   int filter_height)
{
  dim3 threads = makeTileThreads();
  dim3 blocks = makeTileBlocks(width, height);
  size_t shared_bytes = sharedBytesForTile(filter_width, filter_height);
  double start = nowMs();

  cudaCheck(cudaMemcpy(d_input, h_input, image_bytes, cudaMemcpyHostToDevice), "cudaMemcpy tiled constant input");
  gpuConvolutionTiledConstant<<<blocks, threads, shared_bytes>>>(d_input,
                                                                 d_output,
                                                                 width,
                                                                 height,
                                                                 filter_width,
                                                                 filter_height);
  cudaCheck(cudaPeekAtLastError(), "gpuConvolutionTiledConstant launch");
  cudaCheck(cudaDeviceSynchronize(), "gpuConvolutionTiledConstant sync");
  cudaCheck(cudaMemcpy(h_output, d_output, image_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy tiled constant output");

  return nowMs() - start;
}

static void runUntimedVersions(const float *h_input,
                               float *cpu_output,
                               float *global_output,
                               float *tiled_output,
                               float *constant_output,
                               float *tiled_constant_output,
                               float *d_input,
                               float *d_output,
                               float *d_filter,
                               size_t image_bytes,
                               int width,
                               int height,
                               const FilterSpec *filter)
{
  cpuConvolution(h_input,
                 cpu_output,
                 width,
                 height,
                 filter->values,
                 filter->width,
                 filter->height);

  (void) timeGpuGlobal(h_input,
                       global_output,
                       d_input,
                       d_output,
                       d_filter,
                       image_bytes,
                       width,
                       height,
                       filter->width,
                       filter->height);

  (void) timeGpuTiled(h_input,
                      tiled_output,
                      d_input,
                      d_output,
                      d_filter,
                      image_bytes,
                      width,
                      height,
                      filter->width,
                      filter->height);

  (void) timeGpuConstant(h_input,
                         constant_output,
                         d_input,
                         d_output,
                         image_bytes,
                         width,
                         height,
                         filter->width,
                         filter->height);

  (void) timeGpuTiledConstant(h_input,
                              tiled_constant_output,
                              d_input,
                              d_output,
                              image_bytes,
                              width,
                              height,
                              filter->width,
                              filter->height);
}

static void saveExampleOutputs(const char *output_dir,
                               const ImageSpec *image,
                               const FilterSpec *filter,
                               const float *input,
                               const float *cpu_output,
                               int save_input)
{
  char path[512];

  if (save_input)
  {
    snprintf(path, sizeof(path), "%s/sample_%s_input.pgm", output_dir, image->name);
    writePgm(path, input, image->width, image->height);
  }

  snprintf(path, sizeof(path), "%s/sample_%s_%s_cpu.pgm", output_dir, image->name, filter->name);
  writePgm(path, cpu_output, image->width, image->height);
}

static void printHeader(FILE *fp)
{
  fprintf(fp,
          "image,pattern,width,height,pixels,filter,filter_width,filter_height,version,repeat,time_ms,verified\n");
}

static void runOneTest(FILE *fp,
                       const ImageSpec *image,
                       const FilterSpec *filter,
                       int repeats,
                       const char *output_dir,
                       int save_examples)
{
  int count = image->width * image->height;
  size_t image_bytes = (size_t) count * sizeof(float);
  size_t filter_bytes = (size_t) filter->width * (size_t) filter->height * sizeof(float);
  float *h_input;
  float *h_cpu_output;
  float *h_global_output;
  float *h_tiled_output;
  float *h_constant_output;
  float *h_tiled_constant_output;
  float *d_input;
  float *d_output;
  float *d_filter;
  double avg[VERSION_COUNT] = { 0.0, 0.0, 0.0, 0.0, 0.0 };
  double samples[VERSION_COUNT][64] = { 0.0 };
  int verified[VERSION_COUNT] = { 1, 1, 1, 1, 1 };
  int repeat;

  cudaCheck(cudaMallocHost((void **) &h_input, image_bytes), "cudaMallocHost input");
  cudaCheck(cudaMallocHost((void **) &h_cpu_output, image_bytes), "cudaMallocHost cpu output");
  cudaCheck(cudaMallocHost((void **) &h_global_output, image_bytes), "cudaMallocHost global output");
  cudaCheck(cudaMallocHost((void **) &h_tiled_output, image_bytes), "cudaMallocHost tiled output");
  cudaCheck(cudaMallocHost((void **) &h_constant_output, image_bytes), "cudaMallocHost constant output");
  cudaCheck(cudaMallocHost((void **) &h_tiled_constant_output, image_bytes), "cudaMallocHost tiled constant output");

  cudaCheck(cudaMalloc((void **) &d_input, image_bytes), "cudaMalloc input");
  cudaCheck(cudaMalloc((void **) &d_output, image_bytes), "cudaMalloc output");
  cudaCheck(cudaMalloc((void **) &d_filter, filter_bytes), "cudaMalloc filter");

  initImage(h_input, image->width, image->height, image->pattern);
  cudaCheck(cudaMemcpy(d_filter, filter->values, filter_bytes, cudaMemcpyHostToDevice), "cudaMemcpy filter");
  cudaCheck(cudaMemcpyToSymbol(constant_filter, filter->values, filter_bytes), "cudaMemcpyToSymbol filter");

  for (repeat = 1; repeat <= repeats; ++repeat)
  {
    double cpu_time = timeCpu(h_input,
                              h_cpu_output,
                              image->width,
                              image->height,
                              filter->values,
                              filter->width,
                              filter->height);

    double global_time = timeGpuGlobal(h_input,
                                       h_global_output,
                                       d_input,
                                       d_output,
                                       d_filter,
                                       image_bytes,
                                       image->width,
                                       image->height,
                                       filter->width,
                                       filter->height);

    double tiled_time = timeGpuTiled(h_input,
                                     h_tiled_output,
                                     d_input,
                                     d_output,
                                     d_filter,
                                     image_bytes,
                                     image->width,
                                     image->height,
                                     filter->width,
                                     filter->height);

    double constant_time = timeGpuConstant(h_input,
                                           h_constant_output,
                                           d_input,
                                           d_output,
                                           image_bytes,
                                           image->width,
                                           image->height,
                                           filter->width,
                                           filter->height);

    double tiled_constant_time = timeGpuTiledConstant(h_input,
                                                      h_tiled_constant_output,
                                                      d_input,
                                                      d_output,
                                                      image_bytes,
                                                      image->width,
                                                      image->height,
                                                      filter->width,
                                                      filter->height);

    avg[VERSION_CPU] += cpu_time;
    avg[VERSION_GLOBAL] += global_time;
    avg[VERSION_TILED] += tiled_time;
    avg[VERSION_CONSTANT] += constant_time;
    avg[VERSION_TILED_CONSTANT] += tiled_constant_time;

    samples[VERSION_CPU][repeat - 1] = cpu_time;
    samples[VERSION_GLOBAL][repeat - 1] = global_time;
    samples[VERSION_TILED][repeat - 1] = tiled_time;
    samples[VERSION_CONSTANT][repeat - 1] = constant_time;
    samples[VERSION_TILED_CONSTANT][repeat - 1] = tiled_constant_time;
  }

  runUntimedVersions(h_input,
                     h_cpu_output,
                     h_global_output,
                     h_tiled_output,
                     h_constant_output,
                     h_tiled_constant_output,
                     d_input,
                     d_output,
                     d_filter,
                     image_bytes,
                     image->width,
                     image->height,
                     filter);

  verified[VERSION_GLOBAL] = checkSame(h_cpu_output, h_global_output, count);
  verified[VERSION_TILED] = checkSame(h_cpu_output, h_tiled_output, count);
  verified[VERSION_CONSTANT] = checkSame(h_cpu_output, h_constant_output, count);
  verified[VERSION_TILED_CONSTANT] = checkSame(h_cpu_output, h_tiled_constant_output, count);

  for (repeat = 1; repeat <= repeats; ++repeat)
  {
    int k;

    for (k = 0; k < VERSION_COUNT; ++k)
    {
      fprintf(fp,
              "%s,%d,%d,%d,%d,%s,%d,%d,%s,%d,%.6f,%d\n",
              image->name,
              image->pattern,
              image->width,
              image->height,
              count,
              filter->name,
              filter->width,
              filter->height,
              version_names[k],
              repeat,
              samples[k][repeat - 1],
              verified[k]);
    }
  }

  if (save_examples)
  {
    saveExampleOutputs(output_dir, image, filter, h_input, h_cpu_output, filter == &filters[0]);
  }

  printf("%-12s %-18s | cpu %.3f ms | global %.3f ms | tiled %.3f ms | const %.3f ms | tiled const %.3f ms | verify %d %d %d %d\n",
         image->name,
         filter->name,
         avg[VERSION_CPU] / repeats,
         avg[VERSION_GLOBAL] / repeats,
         avg[VERSION_TILED] / repeats,
         avg[VERSION_CONSTANT] / repeats,
         avg[VERSION_TILED_CONSTANT] / repeats,
         verified[VERSION_GLOBAL],
         verified[VERSION_TILED],
         verified[VERSION_CONSTANT],
         verified[VERSION_TILED_CONSTANT]);

  cudaCheck(cudaFree(d_filter), "cudaFree filter");
  cudaCheck(cudaFree(d_output), "cudaFree output");
  cudaCheck(cudaFree(d_input), "cudaFree input");

  cudaCheck(cudaFreeHost(h_tiled_constant_output), "cudaFreeHost tiled constant output");
  cudaCheck(cudaFreeHost(h_constant_output), "cudaFreeHost constant output");
  cudaCheck(cudaFreeHost(h_tiled_output), "cudaFreeHost tiled output");
  cudaCheck(cudaFreeHost(h_global_output), "cudaFreeHost global output");
  cudaCheck(cudaFreeHost(h_cpu_output), "cudaFreeHost cpu output");
  cudaCheck(cudaFreeHost(h_input), "cudaFreeHost input");
}

static void usage(const char *program)
{
  printf("Usage: %s [repeats] [output_dir]\n", program);
  printf("Default repeats: %d\n", DEFAULT_REPEATS);
}

int main(int argc, char **argv)
{
  int repeats = DEFAULT_REPEATS;
  const char *output_dir = "lab_output";
  int i;
  int j;
  char results_path[512];
  char mkdir_command[768];
  FILE *fp;

  if (argc >= 2)
  {
    if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
    {
      usage(argv[0]);
      return 0;
    }

    repeats = atoi(argv[1]);
  }

  if (argc >= 3)
  {
    output_dir = argv[2];
  }

  if (repeats <= 0)
  {
    repeats = DEFAULT_REPEATS;
  }

  if (repeats > 64)
  {
    repeats = 64;
  }

  snprintf(mkdir_command, sizeof(mkdir_command), "mkdir -p %s", output_dir);
  system(mkdir_command);

  snprintf(results_path, sizeof(results_path), "%s/results.csv", output_dir);
  fp = fopen(results_path, "w");

  if (fp == NULL)
  {
    fprintf(stderr, "Could not open %s\n", results_path);
    return 1;
  }

  printHeader(fp);

  printf("Running %d repeats per image/filter/version\n", repeats);
  printf("Writing results to %s\n", results_path);

  for (i = 0; i < IMAGE_COUNT; ++i)
  {
    for (j = 0; j < FILTER_COUNT; ++j)
    {
      int save_examples = (i == 1);
      runOneTest(fp, &images[i], &filters[j], repeats, output_dir, save_examples);
      fflush(fp);
    }
  }

  fclose(fp);

  printf("Done. Next step: python3 analyze_results.py %s/results.csv %s\n", output_dir, output_dir);
  return 0;
}
