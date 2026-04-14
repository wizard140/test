#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace std;

static const int TILE_SIZE = 32;
static const int GPU_BLOCK_X = 32;
static const int GPU_BLOCK_Y = 8;

struct Matrix
{
    int rows;
    int cols;
    vector<float> values;
};

struct DiffStats
{
    double maxAbsDiff;
    double meanAbsDiff;
    double rmse;
};

struct RunResult
{
    double ms;
    double gflops;
};

static inline int idx(int row, int col, int cols)
{
    return row * cols + col;
}

static void fillRandom(Matrix& mat, mt19937& rng)
{
    uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < mat.values.size(); i++)
    {
        mat.values[i] = dist(rng);
    }
}

static bool readMatrix(const string& fileName, Matrix& mat)
{
    ifstream in(fileName);

    if (!in.is_open())
    {
        cerr << "Could not open file: " << fileName << "\n";
        return false;
    }

    in >> mat.rows >> mat.cols;

    if (!in || mat.rows <= 0 || mat.cols <= 0)
    {
        cerr << "Invalid matrix header in file: " << fileName << "\n";
        return false;
    }

    mat.values.assign(static_cast<size_t>(mat.rows) * mat.cols, 0.0f);

    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            in >> mat.values[idx(i, j, mat.cols)];

            if (!in)
            {
                cerr << "Not enough values in file: " << fileName << "\n";
                return false;
            }
        }
    }

    return true;
}

static bool writeMatrix(const string& fileName, const Matrix& mat)
{
    ofstream out(fileName);

    if (!out.is_open())
    {
        cerr << "Could not write file: " << fileName << "\n";
        return false;
    }

    out << mat.rows << " " << mat.cols << "\n";
    out << fixed << setprecision(6);

    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            out << mat.values[idx(i, j, mat.cols)];

            if (j + 1 < mat.cols)
            {
                out << " ";
            }
        }

        out << "\n";
    }

    return true;
}

static bool sameShapeForMultiply(const Matrix& a, const Matrix& b)
{
    return a.cols == b.rows;
}

static double outputSize(const Matrix& a, const Matrix& b)
{
    return static_cast<double>(a.rows) * b.cols;
}

static double totalOps(const Matrix& a, const Matrix& b)
{
    return 2.0 * a.rows * a.cols * b.cols;
}

static void multiplyCpuSingle(const Matrix& a, const Matrix& b, Matrix& c)
{
    c.rows = a.rows;
    c.cols = b.cols;
    c.values.assign(static_cast<size_t>(c.rows) * c.cols, 0.0f);

    for (int i = 0; i < a.rows; i++)
    {
        for (int j = 0; j < b.cols; j++)
        {
            float sum = 0.0f;

            for (int k = 0; k < a.cols; k++)
            {
                sum += a.values[idx(i, k, a.cols)] * b.values[idx(k, j, b.cols)];
            }

            c.values[idx(i, j, c.cols)] = sum;
        }
    }
}

__global__ void multiplyNaiveKernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c,
                                    int m, int n, int p)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p)
    {
        float sum = 0.0f;

        for (int k = 0; k < n; k++)
        {
            sum += a[row * n + k] * b[k * p + col];
        }

        c[row * p + col] = sum;
    }
}

__global__ void multiplyTiledKernel(const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ c,
                                    int m, int n, int p)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int tileCount = (n + TILE_SIZE - 1) / TILE_SIZE;

    for (int tile = 0; tile < tileCount; tile++)
    {
        int aCol = tile * TILE_SIZE + threadIdx.x;
        int bRow = tile * TILE_SIZE + threadIdx.y;

        if (threadIdx.y < TILE_SIZE && threadIdx.x < TILE_SIZE)
        {
            if (row < m && aCol < n)
            {
                tileA[threadIdx.y][threadIdx.x] = a[row * n + aCol];
            }
            else
            {
                tileA[threadIdx.y][threadIdx.x] = 0.0f;
            }

            if (bRow < n && col < p)
            {
                tileB[threadIdx.y][threadIdx.x] = b[bRow * p + col];
            }
            else
            {
                tileB[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; k++)
        {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < p)
    {
        c[row * p + col] = sum;
    }
}

static bool cudaCheck(cudaError_t err, const string& message)
{
    if (err != cudaSuccess)
    {
        cerr << message << " : " << cudaGetErrorString(err) << "\n";
        return false;
    }

    return true;
}

static int chooseGpuRepeats(const Matrix& a, const Matrix& b)
{
    long long work = 1LL * a.rows * a.cols * b.cols;

    if (work < 5000000LL)
    {
        return 100;
    }
    else if (work < 50000000LL)
    {
        return 50;
    }
    else if (work < 200000000LL)
    {
        return 20;
    }

    return 10;
}

static DiffStats compareMatrices(const Matrix& x, const Matrix& y)
{
    DiffStats stats;
    stats.maxAbsDiff = 0.0;
    stats.meanAbsDiff = 0.0;
    stats.rmse = 0.0;

    if (x.rows != y.rows || x.cols != y.cols)
    {
        stats.maxAbsDiff = numeric_limits<double>::infinity();
        stats.meanAbsDiff = numeric_limits<double>::infinity();
        stats.rmse = numeric_limits<double>::infinity();
        return stats;
    }

    double absSum = 0.0;
    double sqSum = 0.0;

    for (size_t i = 0; i < x.values.size(); i++)
    {
        double diff = fabs(static_cast<double>(x.values[i]) - static_cast<double>(y.values[i]));
        stats.maxAbsDiff = max(stats.maxAbsDiff, diff);
        absSum += diff;
        sqSum += diff * diff;
    }

    if (!x.values.empty())
    {
        stats.meanAbsDiff = absSum / x.values.size();
        stats.rmse = sqrt(sqSum / x.values.size());
    }

    return stats;
}

static RunResult runGpuVersion(const Matrix& a, const Matrix& b, Matrix& c, bool tiled)
{
    RunResult result;
    result.ms = -1.0;
    result.gflops = -1.0;

    c.rows = a.rows;
    c.cols = b.cols;
    c.values.assign(static_cast<size_t>(c.rows) * c.cols, 0.0f);

    size_t bytesA = static_cast<size_t>(a.rows) * a.cols * sizeof(float);
    size_t bytesB = static_cast<size_t>(b.rows) * b.cols * sizeof(float);
    size_t bytesC = static_cast<size_t>(c.rows) * c.cols * sizeof(float);

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;
    float* hPinnedA = nullptr;
    float* hPinnedB = nullptr;
    float* hPinnedC = nullptr;

    if (!cudaCheck(cudaHostAlloc(&hPinnedA, bytesA, cudaHostAllocDefault), "Pinned alloc A"))
    {
        return result;
    }

    if (!cudaCheck(cudaHostAlloc(&hPinnedB, bytesB, cudaHostAllocDefault), "Pinned alloc B"))
    {
        cudaFreeHost(hPinnedA);
        return result;
    }

    if (!cudaCheck(cudaHostAlloc(&hPinnedC, bytesC, cudaHostAllocDefault), "Pinned alloc C"))
    {
        cudaFreeHost(hPinnedA);
        cudaFreeHost(hPinnedB);
        return result;
    }

    copy(a.values.begin(), a.values.end(), hPinnedA);
    copy(b.values.begin(), b.values.end(), hPinnedB);

    if (!cudaCheck(cudaMalloc(&dA, bytesA), "Device alloc A") ||
        !cudaCheck(cudaMalloc(&dB, bytesB), "Device alloc B") ||
        !cudaCheck(cudaMalloc(&dC, bytesC), "Device alloc C"))
    {
        if (dA) cudaFree(dA);
        if (dB) cudaFree(dB);
        if (dC) cudaFree(dC);
        cudaFreeHost(hPinnedA);
        cudaFreeHost(hPinnedB);
        cudaFreeHost(hPinnedC);
        return result;
    }

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    dim3 block(GPU_BLOCK_X, GPU_BLOCK_Y);
    dim3 grid((b.cols + block.x - 1) / block.x, (a.rows + block.y - 1) / block.y);

    if (!cudaCheck(cudaMemcpy(dA, hPinnedA, bytesA, cudaMemcpyHostToDevice), "Warmup copy A") ||
        !cudaCheck(cudaMemcpy(dB, hPinnedB, bytesB, cudaMemcpyHostToDevice), "Warmup copy B"))
    {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        cudaFreeHost(hPinnedA);
        cudaFreeHost(hPinnedB);
        cudaFreeHost(hPinnedC);
        return result;
    }

    if (tiled)
    {
        multiplyTiledKernel<<<grid, block>>>(dA, dB, dC, a.rows, a.cols, b.cols);
    }
    else
    {
        multiplyNaiveKernel<<<grid, block>>>(dA, dB, dC, a.rows, a.cols, b.cols);
    }

    if (!cudaCheck(cudaGetLastError(), "Warmup kernel launch") ||
        !cudaCheck(cudaDeviceSynchronize(), "Warmup synchronize"))
    {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        cudaFreeHost(hPinnedA);
        cudaFreeHost(hPinnedB);
        cudaFreeHost(hPinnedC);
        return result;
    }

    int repeats = chooseGpuRepeats(a, b);

    cudaEventRecord(startEvent);

    for (int r = 0; r < repeats; r++)
    {
        if (!cudaCheck(cudaMemcpy(dA, hPinnedA, bytesA, cudaMemcpyHostToDevice), "Timed copy A") ||
            !cudaCheck(cudaMemcpy(dB, hPinnedB, bytesB, cudaMemcpyHostToDevice), "Timed copy B"))
        {
            cudaEventDestroy(startEvent);
            cudaEventDestroy(stopEvent);
            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dC);
            cudaFreeHost(hPinnedA);
            cudaFreeHost(hPinnedB);
            cudaFreeHost(hPinnedC);
            return result;
        }

        if (tiled)
        {
            multiplyTiledKernel<<<grid, block>>>(dA, dB, dC, a.rows, a.cols, b.cols);
        }
        else
        {
            multiplyNaiveKernel<<<grid, block>>>(dA, dB, dC, a.rows, a.cols, b.cols);
        }

        if (!cudaCheck(cudaGetLastError(), "Kernel launch"))
        {
            cudaEventDestroy(startEvent);
            cudaEventDestroy(stopEvent);
            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dC);
            cudaFreeHost(hPinnedA);
            cudaFreeHost(hPinnedB);
            cudaFreeHost(hPinnedC);
            return result;
        }

        if (!cudaCheck(cudaMemcpy(hPinnedC, dC, bytesC, cudaMemcpyDeviceToHost), "Timed copy C"))
        {
            cudaEventDestroy(startEvent);
            cudaEventDestroy(stopEvent);
            cudaFree(dA);
            cudaFree(dB);
            cudaFree(dC);
            cudaFreeHost(hPinnedA);
            cudaFreeHost(hPinnedB);
            cudaFreeHost(hPinnedC);
            return result;
        }
    }

    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);

    result.ms = elapsedMs / repeats;
    result.gflops = totalOps(a, b) / (result.ms / 1000.0) / 1.0e9;
    c.values.assign(hPinnedC, hPinnedC + static_cast<size_t>(c.rows) * c.cols);

    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(hPinnedA);
    cudaFreeHost(hPinnedB);
    cudaFreeHost(hPinnedC);

    return result;
}

static RunResult runCpuVersion(const Matrix& a, const Matrix& b, Matrix& c)
{
    RunResult result;
    long long work = 1LL * a.rows * a.cols * b.cols;
    int repeats = 1;

    if (work < 5000000LL)
    {
        repeats = 200;
    }
    else if (work < 50000000LL)
    {
        repeats = 50;
    }
    else if (work < 200000000LL)
    {
        repeats = 10;
    }
    else
    {
        repeats = 3;
    }

    auto start = chrono::high_resolution_clock::now();

    for (int r = 0; r < repeats; r++)
    {
        multiplyCpuSingle(a, b, c);
    }

    auto stop = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> elapsed = stop - start;
    result.ms = elapsed.count() / repeats;
    result.gflops = totalOps(a, b) / (result.ms / 1000.0) / 1.0e9;

    return result;
}

static void printUsage(const char* progName)
{
    cout << "Usage:\n";
    cout << "  " << progName << " m n p\n";
    cout << "  " << progName << " matrixA.txt matrixB.txt\n\n";
    cout << "Random mode generates A(m x n) and B(n x p).\n";
    cout << "File mode reads A and B, then writes the tiled result to output.txt.\n";
}

static bool parsePositiveInt(const string& s, int& value)
{
    try
    {
        long long temp = stoll(s);

        if (temp <= 0 || temp > numeric_limits<int>::max())
        {
            return false;
        }

        value = static_cast<int>(temp);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

static void printSummary(const Matrix& a, const Matrix& b, const RunResult& cpuResult,
    const RunResult& naiveResult, const RunResult& tiledResult,
    const DiffStats& naiveDiff, const DiffStats& tiledDiff)
{
    cout << fixed << setprecision(4);

    cout << "Matrix A: " << a.rows << " x " << a.cols << "\n";
    cout << "Matrix B: " << b.rows << " x " << b.cols << "\n";
    cout << "Output C: " << a.rows << " x " << b.cols << "\n";
    cout << "Output size (m * p): " << outputSize(a, b) << "\n\n";

    cout << left << setw(18) << "Version"
         << setw(14) << "Time (ms)"
         << setw(14) << "GFLOP/s"
         << "Notes\n";

    cout << left << setw(18) << "CPU single"
         << setw(14) << cpuResult.ms
         << setw(14) << cpuResult.gflops
         << "baseline\n";

    cout << left << setw(18) << "CUDA naive"
         << setw(14) << naiveResult.ms
         << setw(14) << naiveResult.gflops
         << "avg + warmup + H2D/D2H\n";

    cout << left << setw(18) << "CUDA tiled"
         << setw(14) << tiledResult.ms
         << setw(14) << tiledResult.gflops
         << "avg + warmup + H2D/D2H\n";

    cout << "\nConsistency vs CPU\n";
    cout << "Naive max abs diff : " << naiveDiff.maxAbsDiff << "\n";
    cout << "Naive mean abs diff: " << naiveDiff.meanAbsDiff << "\n";
    cout << "Naive RMSE         : " << naiveDiff.rmse << "\n";
    cout << "Tiled max abs diff : " << tiledDiff.maxAbsDiff << "\n";
    cout << "Tiled mean abs diff: " << tiledDiff.meanAbsDiff << "\n";
    cout << "Tiled RMSE         : " << tiledDiff.rmse << "\n";

    cout << "\nSpeedups vs CPU\n";
    cout << "Naive speedup: " << (cpuResult.ms / naiveResult.ms) << "x\n";
    cout << "Tiled speedup: " << (cpuResult.ms / tiledResult.ms) << "x\n";
}

int main(int argc, char* argv[])
{
    if (argc != 3 && argc != 4)
    {
        printUsage(argv[0]);
        return 1;
    }

    Matrix a;
    Matrix b;

    if (argc == 4)
    {
        int m = 0;
        int n = 0;
        int p = 0;

        if (!parsePositiveInt(argv[1], m) ||
            !parsePositiveInt(argv[2], n) ||
            !parsePositiveInt(argv[3], p))
        {
            cerr << "Dimensions must be positive integers.\n";
            return 1;
        }

        a.rows = m;
        a.cols = n;
        a.values.assign(static_cast<size_t>(m) * n, 0.0f);

        b.rows = n;
        b.cols = p;
        b.values.assign(static_cast<size_t>(n) * p, 0.0f);

        mt19937 rng(42);
        fillRandom(a, rng);
        fillRandom(b, rng);
    }
    else
    {
        if (!readMatrix(argv[1], a) || !readMatrix(argv[2], b))
        {
            return 1;
        }

        if (!sameShapeForMultiply(a, b))
        {
            cerr << "Matrix shapes do not match for multiplication.\n";
            return 1;
        }
    }

    Matrix cpuC;
    Matrix naiveC;
    Matrix tiledC;

    RunResult cpuResult = runCpuVersion(a, b, cpuC);
    RunResult naiveResult = runGpuVersion(a, b, naiveC, false);
    RunResult tiledResult = runGpuVersion(a, b, tiledC, true);

    DiffStats naiveDiff = compareMatrices(cpuC, naiveC);
    DiffStats tiledDiff = compareMatrices(cpuC, tiledC);

    printSummary(a, b, cpuResult, naiveResult, tiledResult, naiveDiff, tiledDiff);

    if (argc == 3)
    {
        if (!writeMatrix("output.txt", tiledC))
        {
            return 1;
        }

        cout << "\nWrote tiled CUDA result to output.txt\n";
    }

    return 0;
}
